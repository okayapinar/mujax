"""Bottleneck timings for the same setup `run_experiment` trains.

`profile(experiment)` uses `experiment.config`, `experiment.environment_factory` and
`experiment.num_actors`. The factory may return a single env or a SAME_STEP vector env,
including wrappers. `examples/profile.py` is the Gymnasium-id command line.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import gymnasium as gym
import jax
import numpy as np

from mujax.actor import Actor
from mujax.algorithm import SearchPolicy, ValueFn
from mujax.algorithms import algorithm_for
from mujax.algorithms.muzero import Support, make_targets
from mujax.config import SIZES, MuZeroConfig
from mujax.experiment import ExperimentConfig, make_optimizer, setup_devices
from mujax.learner import Learner, LossFn
from mujax.loop import Counter, EnvironmentLoop
from mujax.observers import ActionFractionObserver, EnvLoopObserver, PolicyEntropyObserver
from mujax.reanalyze import Reanalyzer
from mujax.replay import Buffer, item_bytes
from mujax.types import EnvironmentSpec, TimeStep, as_vector_env, make_environment_spec

EnvironmentFactory = Callable[[int], gym.Env | gym.vector.VectorEnv]


class Timing(NamedTuple):
    """Execution time statistics in milliseconds."""

    mean_ms: float
    p95_ms: float | None = None
    compile_ms: float = 0.0


class ConcurrentResult(NamedTuple):
    """Realized throughput when actor threads and learner run together."""

    actor_steps_per_sec: float
    learner_steps_per_sec: float
    elapsed_sec: float


class ProfileResult(NamedTuple):
    """Collected timings and throughput metrics from `profile(experiment)`."""

    actor: dict[str, Timing]
    learner: dict[str, Timing]
    reanalyze_sub: dict[str, Timing]
    learner_sub: dict[str, Timing]
    concurrent: ConcurrentResult | None


def summarize_samples(samples_ms: list[float], compile_ms: float = 0.0) -> Timing:
    """Computes mean and p95 from a list of millisecond measurements."""
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    p95_ms = float(np.percentile(arr, 95))
    return Timing(mean_ms=mean_ms, p95_ms=p95_ms, compile_ms=compile_ms)


def time_jax_call(fn: Callable[..., Any], *args: Any, iters: int) -> Timing:
    """Measures first-call (compile + run) time, then per-iteration mean and p95 in ms."""
    compile_start = time.perf_counter()
    first_output = fn(*args)
    jax.block_until_ready(first_output)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        output = fn(*args)
        jax.block_until_ready(output)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        samples_ms.append(elapsed_ms)

    return summarize_samples(samples_ms, compile_ms=compile_ms)


def count_parameters(params: Any) -> tuple[int, float]:
    """Returns (total_param_count, total_megabytes) for a parameter tree."""
    leaves = jax.tree.leaves(params)
    num_params = sum(int(leaf.size) for leaf in leaves)
    total_bytes = sum(int(leaf.nbytes) for leaf in leaves)
    total_mb = total_bytes / (1024.0 * 1024.0)
    return num_params, total_mb


def _size_name(config: MuZeroConfig) -> str | None:
    """Preset name when every layer stack matches `SIZES`, otherwise None."""
    layer_fields = [getattr(config, field.name) for field in dataclasses.fields(config) if field.name.endswith("_layer_sizes")]
    if not layer_fields:
        return None
    layers = layer_fields[0]
    if any(other != layers for other in layer_fields):
        return None
    for name, (width, depth, embedding_dim) in SIZES.items():
        if layers == (width,) * depth and config.embedding_dim == embedding_dim:
            return name
    return None


def _env_label(env: gym.vector.VectorEnv, extra: dict[str, Any]) -> str:
    """Name shown in the summary: `extra['env_id']`, else the Gymnasium spec id."""
    named = extra.get("env_id")
    if named:
        return str(named)
    spec = getattr(env, "spec", None)
    env_id = getattr(spec, "id", None)
    if env_id:
        return str(env_id)
    subenvs = getattr(env, "envs", None)
    if subenvs:
        sub_spec = getattr(subenvs[0], "spec", None)
        sub_id = getattr(sub_spec, "id", None)
        if sub_id:
            return str(sub_id)
    return "custom"


def _open_actor_envs(factory: EnvironmentFactory, seed: int, num_actors: int) -> tuple[list[gym.vector.VectorEnv], list[int]]:
    """One SAME_STEP vector env per actor, seeded like `run_experiment`."""
    first = as_vector_env(factory(seed))
    num_envs = int(first.num_envs)
    seeds = [seed + index * num_envs for index in range(num_actors)]
    envs = [first]
    for actor_seed in seeds[1:]:
        envs.append(as_vector_env(factory(actor_seed)))
    same_width = all(int(env.num_envs) == num_envs for env in envs)
    if same_width:
        return envs, seeds
    for env in envs:
        env.close(terminate=True)
    raise ValueError("environment_factory her aktor icin ayni num_envs ile env uretmeli")


def _close_envs(envs: list[gym.vector.VectorEnv]) -> None:
    for env in envs:
        env.close(terminate=True)


def _make_observers(observer_factories: Sequence[Callable[[], EnvLoopObserver]]) -> list[EnvLoopObserver]:
    custom_observers = [factory() for factory in observer_factories]
    return [ActionFractionObserver(), PolicyEntropyObserver(), *custom_observers]


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f} sn"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f} dk"
    hours = minutes / 60.0
    return f"{hours:.2f} saat"


def print_system_summary(
    algo_name: str,
    env_label: str,
    num_envs: int,
    num_actors: int,
    spec: EnvironmentSpec,
    config: MuZeroConfig,
    params: Any,
    actor_device: jax.Device,
    learner_device: jax.Device,
) -> None:
    """Prints hardware, model size, and replay/batch memory footprint."""
    num_params, param_mb = count_parameters(params)
    step_bytes = item_bytes(spec)
    replay_gb = (config.max_replay_size * step_bytes) / (1024.0**3)
    batch_transitions = config.batch_size * config.sequence_length
    batch_mb = (batch_transitions * step_bytes) / (1024.0**2)

    reanalyze_batch = round(config.batch_size * config.reanalyze_ratio)
    search_steps = min(config.num_unroll_steps + 1, config.sequence_length)
    reanalyze_searches = reanalyze_batch * search_steps
    cpu_cores = os.cpu_count() or 1
    total_envs = num_actors * num_envs

    size_name = _size_name(config)
    size_label = f"size={size_name}, " if size_name else ""

    print("\n=== SISTEM VE MODEL OZETI ===")
    print(f"  Algoritma / Boyut  : {algo_name.upper()} ({size_label}embedding_dim={config.embedding_dim})")
    print(f"  Parametre sayisi   : {num_params:,} ({param_mb:.2f} MB)")
    print(f"  Ortam              : {env_label} (obs_dim={spec.obs_dim}, actions={spec.num_actions})")
    print(f"  Aktor yapisi       : {num_actors} thread x {num_envs} vec_env = {total_envs} paralel env ({cpu_cores} CPU cekirdegi)")
    print(f"  Cihazlar           : actor={actor_device} | learner={learner_device}")
    print(
        f"  Batch / Unroll     : batch={config.batch_size}, seq_len={config.sequence_length} (K={config.num_unroll_steps}, n_td={config.num_bootstrapping})"
    )
    print(f"  MCTS / Reanalyze   : sims={config.num_simulations}, reanalyze={reanalyze_batch}/{config.batch_size} ({reanalyze_searches} MCTS/step)")
    print(f"  Bellek izi         : batch={batch_mb:.2f} MB | tam replay ({config.max_replay_size:,} adim)={replay_gb:.2f} GB")


def profile_actor(
    environment_factory: EnvironmentFactory,
    seed: int,
    num_envs: int,
    spec: EnvironmentSpec,
    config: MuZeroConfig,
    policy: SearchPolicy,
    value_fn: ValueFn,
    params: Any,
    device: jax.Device,
    iters: int,
    *,
    learner_device: jax.Device | None = None,
    observer_factories: Sequence[Callable[[], EnvLoopObserver]] = (),
) -> dict[str, Timing]:
    """Measures each part of a single vector-env step on the actor."""
    env = as_vector_env(environment_factory(seed))
    opened = int(env.num_envs)
    if opened != num_envs:
        env.close(terminate=True)
        raise ValueError(f"environment_factory {num_envs} env beklerken {opened} dondurdu")
    min_chunks_steps = num_envs * config.sequence_length * 10
    iter_steps = num_envs * (iters + config.sequence_length * 2)
    max_replay_size = max(min_chunks_steps, iter_steps)
    replay = Buffer(
        spec,
        num_envs=num_envs,
        max_size=max_replay_size,
        sample_batch_size=config.batch_size,
        sequence_length=config.sequence_length,
    )
    adder = replay.adder(0)
    rng_key = jax.random.PRNGKey(0)
    source_device = learner_device if learner_device is not None else device
    learner_params = jax.device_put(params, source_device)
    learner_step = 0

    def get_params() -> tuple[Any, int]:
        return learner_params, learner_step

    actor = Actor(
        policy,
        rng_key,
        get_params,
        num_actions=spec.num_actions,
        replay=adder,
        value_fn=value_fn,
        device=device,
        update_period=config.variable_update_period,
    )
    observers = _make_observers(observer_factories)

    obs, info = env.reset(seed=seed)
    actor.observe_first(obs, info)
    for observer in observers:
        observer.observe_first(env, num_envs)

    # Warmup: run a full sequence_length chunk so both MCTS search and replay chunk write compile
    t_compile_search = 0.0
    t_compile_observe = 0.0
    for step_idx in range(config.sequence_length):
        t0 = time.perf_counter()
        action = actor.select_action(obs)
        t1 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(action)
        last_search = actor.last_search()
        step = TimeStep(obs, action, reward, terminated, truncated, info, last_search)
        t2 = time.perf_counter()
        actor.observe(step)
        jax.block_until_ready(replay._state)
        t3 = time.perf_counter()
        if step_idx == 0:
            t_compile_search = (t1 - t0) * 1000.0
        if step_idx == config.sequence_length - 1:
            t_compile_observe = (t3 - t2) * 1000.0
        obs = next_obs

    dummy_obs = np.zeros((num_envs, spec.obs_dim), dtype=np.float32)
    actor_params = actor._params[0]
    value_out = actor._value(actor_params, dummy_obs)
    jax.block_until_ready(value_out)

    update_period = max(1, config.variable_update_period)
    actor._num_calls = 0
    samples: dict[str, list[float]] = {
        "param_sync (device_put)": [],
        "select_action (MCTS)": [],
        "env.step": [],
        "actor.observe (replay)": [],
        "observers": [],
    }
    for step_idx in range(iters):
        t0 = time.perf_counter()
        if (step_idx + 1) % update_period == 0:
            learner_step += 1
            actor._pull_params()
            jax.block_until_ready(actor._params[0])
        t1 = time.perf_counter()
        action = actor.select_action(obs)
        t2 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(action)
        t3 = time.perf_counter()
        last_search = actor.last_search()
        step = TimeStep(obs, action, reward, terminated, truncated, info, last_search)
        actor.observe(step)
        jax.block_until_ready(replay._state)
        t4 = time.perf_counter()
        for observer in observers:
            observer.observe(env, step)
        t5 = time.perf_counter()

        samples["param_sync (device_put)"].append((t1 - t0) * 1000.0)
        samples["select_action (MCTS)"].append((t2 - t1) * 1000.0)
        samples["env.step"].append((t3 - t2) * 1000.0)
        samples["actor.observe (replay)"].append((t4 - t3) * 1000.0)
        samples["observers"].append((t5 - t4) * 1000.0)
        obs = next_obs

    env.close(terminate=True)
    compile_times = {
        "select_action (MCTS)": t_compile_search,
        "actor.observe (replay)": t_compile_observe,
    }
    return {name: summarize_samples(values, compile_ms=compile_times.get(name, 0.0)) for name, values in samples.items()}


def profile_learner(
    spec: EnvironmentSpec,
    config: MuZeroConfig,
    policy: SearchPolicy,
    value_fn: ValueFn,
    loss_fn: LossFn,
    params: Any,
    device: jax.Device,
    iters: int,
) -> tuple[dict[str, Timing], dict[str, Timing], dict[str, Timing]]:
    """Measures Learner pipeline stages, Reanalyze sub-parts, and Learner gradient_step sub-parts."""
    seq_len = config.sequence_length
    min_chunks = 2
    # Width of the dummy buffer. The timed work is a `batch_size` sample, not this row count.
    num_envs = 16

    max_replay_size = max(config.batch_size * 4, num_envs * seq_len * 10)
    replay = Buffer(
        spec,
        num_envs=num_envs,
        max_size=max_replay_size,
        sample_batch_size=config.batch_size,
        sequence_length=seq_len,
    )
    zero_step = replay._state.experience
    single_step = jax.tree.map(lambda x: np.zeros((num_envs, *x.shape[2:]), dtype=x.dtype), zero_step)
    adder = replay.adder(0)
    for _ in range(seq_len * min_chunks):
        adder.add(single_step)

    optimizer = make_optimizer(config)
    learner = Learner(loss_fn, params, optimizer, device=device)
    reanalyze_key = jax.random.PRNGKey(0)
    stop_event = threading.Event()
    reanalyzer = Reanalyzer(
        replay,
        config=config,
        policy=policy,
        value_fn=value_fn,
        get_params=learner.get_params,
        device=device,
        random_key=reanalyze_key,
        stop_event=stop_event,
    )

    sample_key = jax.random.PRNGKey(1)
    search_key = jax.random.PRNGKey(2)
    raw_sample = replay.sample(sample_key)
    gpu_sample = jax.device_put(raw_sample, device)
    learner_params, learner_steps = learner.get_params()

    t_sample = time_jax_call(replay.sample, sample_key, iters=iters)
    t_transfer = time_jax_call(lambda s: jax.device_put(s, device), raw_sample, iters=iters)
    if reanalyzer._num_reanalyze > 0:
        t_reanalyze = time_jax_call(
            reanalyzer._refresh,
            learner_params,
            gpu_sample.experience,
            search_key,
            learner_steps,
            iters=iters,
        )
    else:
        t_reanalyze = Timing(0.0, 0.0, 0.0)

    t_grad_step = time_jax_call(
        learner._gradient_step,
        learner._params,
        learner._opt_state,
        gpu_sample,
        iters=iters,
    )

    # Reanalyze sub-breakdown: MCTS search vs tail value_fn
    reanalyze_sub: dict[str, Timing] = {}
    if reanalyzer._num_reanalyze > 0:
        n = reanalyzer._num_reanalyze
        k = reanalyzer._num_search
        obs = gpu_sample.experience.observation[:n]
        total_t = obs.shape[1]
        search_obs = obs[:, :k].reshape(n * k, -1)
        invalid = gpu_sample.experience.invalid_actions[:n, :k].reshape(n * k, -1)

        jitted_policy = jax.jit(policy)
        t_mcts = time_jax_call(jitted_policy, learner_params, search_obs, search_key, invalid, learner_steps, iters=iters)
        reanalyze_sub[f"MCTS search ({n}x{k}={n * k})"] = t_mcts

        if total_t > k:
            tail_count = n * (total_t - k)
            tail_obs = obs[:, k:].reshape(tail_count, -1)
            jitted_value = jax.jit(value_fn)
            t_tail = time_jax_call(jitted_value, learner_params, tail_obs, iters=iters)
            reanalyze_sub[f"tail value_fn ({n}x{total_t - k}={tail_count})"] = t_tail

    # Learner gradient_step sub-breakdown: targets vs forward loss vs backward+optimizer
    support = Support.from_config(config)
    jitted_targets = jax.jit(lambda exp: make_targets(config, support, exp))
    jitted_forward = jax.jit(loss_fn)
    t_targets = time_jax_call(jitted_targets, gpu_sample.experience, iters=iters)
    t_forward = time_jax_call(jitted_forward, learner._params, gpu_sample, iters=iters)

    forward_only_mean = max(0.0, t_forward.mean_ms - t_targets.mean_ms)
    backward_only_mean = max(0.0, t_grad_step.mean_ms - t_forward.mean_ms)

    stages = {
        "replay.sample": t_sample,
        "device_put (to learner)": t_transfer,
        "reanalyze (MCTS + tail)": t_reanalyze,
        "learner.gradient_step": t_grad_step,
    }
    learner_sub = {
        "make_targets (TD + HL-Gauss)": Timing(t_targets.mean_ms, t_targets.p95_ms),
        "forward (root + K-step unroll)": Timing(forward_only_mean),
        "backward + optimizer + norms": Timing(backward_only_mean),
    }
    return stages, reanalyze_sub, learner_sub


def profile_concurrent(
    environment_factory: EnvironmentFactory,
    seed: int,
    num_envs: int,
    num_actors: int,
    spec: EnvironmentSpec,
    config: MuZeroConfig,
    policy: SearchPolicy,
    value_fn: ValueFn,
    loss_fn: LossFn,
    params: Any,
    actor_device: jax.Device,
    learner_device: jax.Device,
    duration_sec: float,
    *,
    observer_factories: Sequence[Callable[[], EnvLoopObserver]] = (),
) -> ConcurrentResult:
    """Runs actor threads and learner+reanalyze concurrently to measure real contention."""
    envs, seeds = _open_actor_envs(environment_factory, seed, num_actors)
    opened = int(envs[0].num_envs)
    if opened != num_envs:
        _close_envs(envs)
        raise ValueError(f"environment_factory {num_envs} env beklerken {opened} dondurdu")
    seq_len = config.sequence_length
    num_rows = num_envs * num_actors
    max_replay_size = max(config.batch_size * 4, num_rows * seq_len * 20)
    stop_event = threading.Event()
    counter = Counter()

    # Disable replay_ratio limiter so we measure raw concurrent capacity of both sides
    replay = Buffer(
        spec,
        num_envs=num_envs,
        max_size=max_replay_size,
        min_size=num_rows * seq_len,
        sample_batch_size=config.batch_size,
        sequence_length=seq_len,
        replay_ratio=None,
        stop_event=stop_event,
        num_actors=num_actors,
    )

    optimizer = make_optimizer(config)
    learner = Learner(loss_fn, params, optimizer, device=learner_device, counter=counter, replay=replay)
    dataset = Reanalyzer(
        replay,
        config=config,
        policy=policy,
        value_fn=value_fn,
        get_params=learner.get_params,
        device=learner_device,
        random_key=jax.random.PRNGKey(10),
        stop_event=stop_event,
    )

    loops: list[EnvironmentLoop] = []
    for idx, env in enumerate(envs):
        actor_key = jax.random.PRNGKey(100 + idx)
        adder = replay.adder(idx)
        actor = Actor(
            policy,
            actor_key,
            learner.get_params,
            num_actions=spec.num_actions,
            replay=adder,
            value_fn=value_fn,
            device=actor_device,
            update_period=config.variable_update_period,
        )
        observers = _make_observers(observer_factories)
        loop = EnvironmentLoop(
            env,
            actor,
            name="actor",
            counter=counter,
            observers=observers,
            stop_event=stop_event,
            seed=seeds[idx],
        )
        loops.append(loop)

    # Warmup actors (1 full chunk) and learner (1 step) before starting concurrent timer
    obs_list = [loop._start() for loop in loops]
    for _ in range(seq_len):
        for idx, loop in enumerate(loops):
            obs_list[idx] = loop._step(obs_list[idx])

    first_batch = next(dataset)
    learner.step(first_batch)
    jax.block_until_ready(learner._params)

    start_counts = counter.get()
    start_actor_steps = start_counts.get("actor_steps", 0)
    start_learner_steps = learner.learn_steps

    def run_actor_loop(loop: EnvironmentLoop, initial_obs: np.ndarray) -> None:
        obs = initial_obs
        while not stop_event.is_set():
            obs = loop._step(obs)

    threads = [threading.Thread(target=run_actor_loop, args=(loop, obs_list[idx]), daemon=True) for idx, loop in enumerate(loops)]
    t0 = time.perf_counter()
    for thread in threads:
        thread.start()

    while (time.perf_counter() - t0) < duration_sec:
        batch = next(dataset)
        learner.step(batch)
        jax.block_until_ready(learner._params)

    stop_event.set()
    elapsed = time.perf_counter() - t0
    for thread in threads:
        thread.join(timeout=5.0)
    _close_envs(envs)

    end_counts = counter.get()
    actor_steps = end_counts.get("actor_steps", 0) - start_actor_steps
    learner_steps = learner.learn_steps - start_learner_steps
    return ConcurrentResult(
        actor_steps_per_sec=actor_steps / max(1e-6, elapsed),
        learner_steps_per_sec=learner_steps / max(1e-6, elapsed),
        elapsed_sec=elapsed,
    )


def print_breakdown(title: str, items: dict[str, Timing]) -> float:
    """Prints a formatted timing breakdown table and returns the sum of mean_ms."""
    total_mean = sum(item.mean_ms for item in items.values())
    total_compile = sum(item.compile_ms for item in items.values())
    compile_info = f", JIT derleme: {total_compile:.0f} ms" if total_compile > 0 else ""
    print(f"\n--- {title} (toplam: {total_mean:.2f} ms{compile_info}) ---")
    for name, timing in items.items():
        pct = (timing.mean_ms / total_mean * 100.0) if total_mean > 0 else 0.0
        bar = "#" * int(pct // 4)
        p95_str = f"p95: {timing.p95_ms:7.2f} ms | " if timing.p95_ms is not None else " " * 17
        print(f"  {name:34s}: {timing.mean_ms:8.2f} ms  ({p95_str}{pct:5.1f}%)  {bar}")
    return total_mean


def print_diagnosis(
    config: MuZeroConfig,
    num_envs: int,
    num_actors: int,
    actor_ms: float,
    learner_stages: dict[str, Timing],
    learner_device: jax.Device,
    concurrent: ConcurrentResult | None,
    max_num_learner_steps: int | None = None,
) -> None:
    """Analyzes isolated vs concurrent throughput and suggests actionable fixes."""
    learner_ms = sum(item.mean_ms for item in learner_stages.values())
    single_actor_throughput = (num_envs / (actor_ms / 1000.0)) if actor_ms > 0 else 0.0
    ideal_actor_throughput = single_actor_throughput * num_actors
    ideal_learner_rate = 1000.0 / learner_ms if learner_ms > 0 else 0.0

    print("\n=== UCTAN UCA AKIS & DARBOGAZ ANALIZI ===")
    print(f"  Tek aktor hizi ({num_envs} env, izole)       : {single_actor_throughput:8.1f} env step/sn")
    print(f"  Ideal toplam aktor ({num_actors} aktor, izole) : {ideal_actor_throughput:8.1f} env step/sn")
    print(f"  Ideal Learner + Reanalyze (izole)   : {ideal_learner_rate:8.2f} learner step/sn ({learner_ms:.1f} ms/step)")

    effective_actor_throughput = ideal_actor_throughput
    effective_learner_rate = ideal_learner_rate
    if concurrent is not None:
        actor_efficiency = (concurrent.actor_steps_per_sec / max(1e-6, ideal_actor_throughput)) * 100.0
        learner_efficiency = (concurrent.learner_steps_per_sec / max(1e-6, ideal_learner_rate)) * 100.0
        print(f"\n  Eszamanli gercek hiz ({concurrent.elapsed_sec:.1f} sn test):")
        print(f"  - Gercek aktor hizi                 : {concurrent.actor_steps_per_sec:8.1f} env step/sn (izole hizin %{actor_efficiency:.0f}'i)")
        print(
            f"  - Gercek learner hizi               : {concurrent.learner_steps_per_sec:8.2f} learner step/sn (izole hizin %{learner_efficiency:.0f}'i)"
        )
        effective_actor_throughput = concurrent.actor_steps_per_sec
        effective_learner_rate = concurrent.learner_steps_per_sec

    if learner_device.platform == "cpu":
        print("\n  [!] UYARI: Learner CPU uzerinde calisiyor! Aktor thread'leri ile Learner ayni CPU cekirdeklerini paylasiyor.")
        print('      GPU kullanmak icin: pip install "mujax[cuda12]"')

    realized_learner_rate = effective_learner_rate
    if config.replay_ratio is not None:
        required_env_steps = config.batch_size / config.replay_ratio
        actor_fed_learner_rate = effective_actor_throughput / required_env_steps
        realized_learner_rate = min(effective_learner_rate, actor_fed_learner_rate)
        print(f"\n  Hedef replay_ratio={config.replay_ratio} (1 learner adimi icin {required_env_steps:.0f} env adimi gerekli):")
        print(f"  - Aktorlerin besleyebildigi hiz     : {actor_fed_learner_rate:8.2f} learner step/sn")
        print(f"  - Learner'in isleyebildigi hiz      : {effective_learner_rate:8.2f} learner step/sn")

        if actor_fed_learner_rate < effective_learner_rate * 0.85:
            ratio = effective_learner_rate / max(1e-6, actor_fed_learner_rate)
            needed_actors = max(1, round(num_actors * ratio))
            idle_pct = (1.0 - actor_fed_learner_rate / max(1e-6, effective_learner_rate)) * 100.0
            print(
                f"\n  -> DARBOGAZ: AKTOR (Veri Uretimi). Learner zamaninin ~%{idle_pct:.0f}'ini"
                f" aktorleri bekleyerek geciriyor.\n"
                f"     Cozum: num_actors={needed_actors} yap, paralel env sayisini artir, ya da replay_ratio degerini yukselt."
            )
        elif effective_learner_rate < actor_fed_learner_rate * 0.85:
            reanalyze_ms = learner_stages["reanalyze (MCTS + tail)"].mean_ms
            reanalyze_pct = (reanalyze_ms / learner_ms * 100.0) if learner_ms > 0 else 0.0
            wait_pct = (1.0 - effective_learner_rate / max(1e-6, actor_fed_learner_rate)) * 100.0
            print(f"\n  -> DARBOGAZ: LEARNER / REANALYZE. Aktorler zamanin ~%{wait_pct:.0f}'inde learner'i bekliyor.")
            if reanalyze_pct > 50.0:
                print(
                    f"     En buyuk pay Reanalyze MCTS aramasinda (%{reanalyze_pct:.0f}).\n"
                    f"     Cozum: `reanalyze_ratio` ({config.reanalyze_ratio}) veya `num_simulations` ({config.num_simulations}) dusur."
                )
            else:
                print("     Cozum: agi kucult (with_size) veya num_unroll_steps degerini dusur.")
        else:
            print("\n  -> DENGELI: Aktor veri uretim hizi ile Learner isleme hizi birbirine yakin.")

    num_rows = num_envs * num_actors
    max_length = config.max_replay_size // num_rows
    min_chunks = -(-config.min_replay_size // num_rows)
    min_length = min(max(min_chunks, config.sequence_length), max_length)
    warmup_steps = min_length * num_rows
    warmup_sec = warmup_steps / max(1e-6, effective_actor_throughput)
    warmup_str = _format_duration(warmup_sec)
    print("\n  Tahmini sureler:")
    print(f"  - Replay warmup ({warmup_steps:,} env adimi) : {warmup_str}")
    if max_num_learner_steps is not None and max_num_learner_steps > 0:
        train_sec = max_num_learner_steps / max(1e-6, realized_learner_rate)
        train_str = _format_duration(train_sec)
        total_str = _format_duration(warmup_sec + train_sec)
        print(f"  - Egitim ({max_num_learner_steps:,} learner adimi)      : {train_str} (toplam: {total_str})")


def profile(
    experiment: ExperimentConfig,
    *,
    actor_iters: int = 100,
    learner_iters: int = 20,
    concurrent_seconds: float = 2.0,
) -> ProfileResult:
    """Prints where time goes for the setup `run_experiment(experiment)` would train and returns the timings.

    The factory must return a new env on every call (it is opened once per measured actor, same as training).
    `actor_iters` and `learner_iters` are timed repeats after the first JIT compile.
    `concurrent_seconds` is how long actor threads and the learner run together; 0 skips that part.
    """
    config = experiment.config
    num_actors = int(experiment.num_actors)
    if num_actors < 1:
        raise ValueError(f"num_actors pozitif olmali, gelen: {num_actors}")
    factory = experiment.environment_factory
    seed = experiment.seed

    algo = algorithm_for(config)
    actor_device, learner_device = setup_devices()

    probe = as_vector_env(factory(seed))
    try:
        spec = make_environment_spec(probe)
        num_envs = int(probe.num_envs)
        env_label = _env_label(probe, experiment.extra)
    finally:
        probe.close(terminate=True)

    init_key = jax.random.PRNGKey(seed)
    graphdef, params = algo.init(spec, config, init_key)
    policy = algo.make_policy(graphdef, spec, config, False)
    value_fn = algo.make_value_fn(graphdef, config)
    loss_fn = functools.partial(algo.loss, graphdef, config)

    print_system_summary(algo.name, env_label, num_envs, num_actors, spec, config, params, actor_device, learner_device)

    actor_items = profile_actor(
        factory,
        seed,
        num_envs,
        spec,
        config,
        policy,
        value_fn,
        params,
        actor_device,
        actor_iters,
        learner_device=learner_device,
        observer_factories=experiment.observer_factories,
    )
    actor_ms = print_breakdown(f"AKTOR ADIMI ({num_envs} env, {actor_device})", actor_items)

    learner_stages, reanalyze_sub, learner_sub = profile_learner(spec, config, policy, value_fn, loss_fn, params, learner_device, learner_iters)
    print_breakdown(f"LEARNER + REANALYZE ADIMI (batch={config.batch_size}, {learner_device})", learner_stages)
    if reanalyze_sub:
        print_breakdown("REANALYZE IC DAGILIMI", reanalyze_sub)
    print_breakdown("LEARNER GRADIENT_STEP IC DAGILIMI", learner_sub)

    concurrent_result = None
    if concurrent_seconds > 0:
        concurrent_result = profile_concurrent(
            factory,
            seed,
            num_envs,
            num_actors,
            spec,
            config,
            policy,
            value_fn,
            loss_fn,
            params,
            actor_device,
            learner_device,
            concurrent_seconds,
            observer_factories=experiment.observer_factories,
        )

    print_diagnosis(
        config,
        num_envs,
        num_actors,
        actor_ms,
        learner_stages,
        learner_device,
        concurrent_result,
        max_num_learner_steps=experiment.max_num_learner_steps,
    )
    return ProfileResult(
        actor=actor_items,
        learner=learner_stages,
        reanalyze_sub=reanalyze_sub,
        learner_sub=learner_sub,
        concurrent=concurrent_result,
    )

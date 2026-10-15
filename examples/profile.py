"""Profile MuZero (GMZ / SMZ) components and pinpoint training bottlenecks.

Usage:
    python examples/profile.py
    python examples/profile.py smz --size S --batch-size 512 --num-actors 4
"""

from __future__ import annotations

import dataclasses
import functools
import threading
import time
from typing import Annotated, Any

import jax
import numpy as np
import tyro

from mujax.actor import Actor
from mujax.algorithms import GMZConfig, SMZConfig, algorithm_for
from mujax.algorithms.muzero import Support, make_targets
from mujax.cli import make_vec_env
from mujax.config import MuZeroConfig
from mujax.experiment import make_optimizer, setup_devices
from mujax.learner import Learner
from mujax.observers import ActionFractionObserver, PolicyEntropyObserver
from mujax.reanalyze import Reanalyzer
from mujax.replay import Buffer
from mujax.types import EnvironmentSpec, TimeStep, make_environment_spec

AgentConfig = tyro.conf.OmitSubcommandPrefixes[
    Annotated[GMZConfig, tyro.conf.subcommand("gmz", prefix_name=False)] | Annotated[SMZConfig, tyro.conf.subcommand("smz", prefix_name=False)]
]


@dataclasses.dataclass
class ProfileArgs:
    """Bottleneck profiler for mujax."""

    agent: AgentConfig = dataclasses.field(default_factory=GMZConfig)
    env: str = "CartPole-v1"
    num_envs: int = 16
    num_actors: int = 1
    size: str = "S"
    actor_iters: int = 100
    learner_iters: int = 20


def time_jax_call(fn, *args, iters: int) -> float:
    """Runs warmup (compile) then returns average execution time in milliseconds."""
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        output = fn(*args)
        jax.block_until_ready(output)
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def profile_actor(
    env_id: str,
    num_envs: int,
    spec: EnvironmentSpec,
    config: MuZeroConfig,
    graphdef: Any,
    params: Any,
    device: jax.Device,
    iters: int,
) -> dict[str, float]:
    """Measures each part of a single vector-env step on the actor."""
    algorithm = algorithm_for(config)
    policy = algorithm.make_policy(graphdef, spec, config, False)
    value_fn = algorithm.make_value_fn(graphdef, config)

    env = make_vec_env(env_id, num_envs)
    replay = Buffer(
        spec,
        num_envs=num_envs,
        max_size=num_envs * config.sequence_length * 10,
        sample_batch_size=config.batch_size,
        sequence_length=config.sequence_length,
    )
    actor = Actor(
        policy,
        jax.random.PRNGKey(0),
        lambda: (params, 0),
        num_actions=spec.num_actions,
        replay=replay.adder(0),
        value_fn=value_fn,
        device=device,
        update_period=config.variable_update_period,
    )
    observers = [ActionFractionObserver(), PolicyEntropyObserver()]

    obs, info = env.reset(seed=0)
    actor.observe_first(obs, info)
    for observer in observers:
        observer.observe_first(env, num_envs)

    # Warmup JIT (policy search and bootstrap value_fn)
    for _ in range(5):
        action = actor.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        step = TimeStep(obs, action, reward, terminated, truncated, info, actor.last_search())
        actor.observe(step)
        obs = next_obs

    dummy_obs = np.zeros((num_envs, spec.obs_dim), dtype=np.float32)
    jax.block_until_ready(actor._value(actor._params[0], dummy_obs))

    timings = {"select_action (MCTS)": 0.0, "env.step": 0.0, "actor.observe (replay)": 0.0, "observers": 0.0}
    for _ in range(iters):
        t0 = time.perf_counter()
        action = actor.select_action(obs)
        t1 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(action)
        t2 = time.perf_counter()
        step = TimeStep(obs, action, reward, terminated, truncated, info, actor.last_search())
        actor.observe(step)
        t3 = time.perf_counter()
        for observer in observers:
            observer.observe(env, step)
        t4 = time.perf_counter()

        timings["select_action (MCTS)"] += t1 - t0
        timings["env.step"] += t2 - t1
        timings["actor.observe (replay)"] += t3 - t2
        timings["observers"] += t4 - t3
        obs = next_obs

    env.close()
    return {name: (seconds / iters) * 1000.0 for name, seconds in timings.items()}


def profile_learner(
    spec: EnvironmentSpec,
    config: MuZeroConfig,
    graphdef: Any,
    params: Any,
    device: jax.Device,
    iters: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measures Reanalyzer and Learner stages, plus sub-components inside Learner."""
    algorithm = algorithm_for(config)
    seq_len = config.sequence_length
    min_chunks = 2
    num_envs = 16

    replay = Buffer(
        spec,
        num_envs=num_envs,
        max_size=max(config.batch_size * 4, num_envs * seq_len * 10),
        sample_batch_size=config.batch_size,
        sequence_length=seq_len,
    )
    zero_step = replay._state.experience
    single_step = jax.tree.map(lambda x: np.zeros((num_envs, *x.shape[2:]), dtype=x.dtype), zero_step)
    for _ in range(seq_len * min_chunks):
        replay.adder(0).add(single_step)

    loss_fn = functools.partial(algorithm.loss, graphdef, config)
    optimizer = make_optimizer(config)
    learner = Learner(loss_fn, params, optimizer, device=device)
    reanalyzer = Reanalyzer(
        replay,
        config=config,
        policy=algorithm.make_policy(graphdef, spec, config, False),
        value_fn=algorithm.make_value_fn(graphdef, config),
        get_params=learner.get_params,
        device=device,
        random_key=jax.random.PRNGKey(0),
        stop_event=threading.Event(),
    )

    sample_key = jax.random.PRNGKey(1)
    search_key = jax.random.PRNGKey(2)
    raw_sample = replay.sample(sample_key)
    gpu_sample = jax.device_put(raw_sample, device)
    learner_params, learner_steps = learner.get_params()

    t_sample = time_jax_call(replay.sample, sample_key, iters=iters)
    t_transfer = time_jax_call(lambda s: jax.device_put(s, device), raw_sample, iters=iters)
    t_reanalyze = (
        time_jax_call(reanalyzer._refresh, learner_params, gpu_sample.experience, search_key, learner_steps, iters=iters)
        if reanalyzer._num_reanalyze > 0
        else 0.0
    )
    t_grad_step = time_jax_call(learner._gradient_step, learner._params, learner._opt_state, gpu_sample, iters=iters)

    # Sub-breakdown of Learner step: targets vs forward loss vs backward+optimizer
    support = Support.from_config(config)
    jitted_targets = jax.jit(lambda exp: make_targets(config, support, exp))
    jitted_forward = jax.jit(loss_fn)
    t_targets = time_jax_call(jitted_targets, gpu_sample.experience, iters=iters)
    t_forward = time_jax_call(jitted_forward, learner._params, gpu_sample, iters=iters)

    stages = {
        "replay.sample": t_sample,
        "device_put (to learner)": t_transfer,
        "reanalyze (MCTS + tail)": t_reanalyze,
        "learner.gradient_step": t_grad_step,
    }
    learner_sub = {
        "make_targets (TD + HL-Gauss)": t_targets,
        "forward (root + K-step unroll)": max(0.0, t_forward - t_targets),
        "backward + optimizer + norms": max(0.0, t_grad_step - t_forward),
    }
    return stages, learner_sub


def print_breakdown(title: str, items: dict[str, float]) -> float:
    total = sum(items.values())
    print(f"\n--- {title} (toplam: {total:.2f} ms) ---")
    for name, ms in items.items():
        pct = (ms / total * 100.0) if total > 0 else 0.0
        bar = "#" * int(pct // 4)
        print(f"  {name:32s}: {ms:8.2f} ms  ({pct:5.1f}%)  {bar}")
    return total


def print_diagnosis(
    config: MuZeroConfig,
    num_envs: int,
    num_actors: int,
    actor_ms: float,
    learner_stages: dict[str, float],
    learner_device: jax.Device,
) -> None:
    learner_ms = sum(learner_stages.values())
    single_actor_throughput = (num_envs / (actor_ms / 1000.0)) if actor_ms > 0 else 0.0
    total_actor_throughput = single_actor_throughput * num_actors
    max_learner_steps_per_sec = 1000.0 / learner_ms if learner_ms > 0 else 0.0

    print("\n=== UCTAN UCA AKIS & DARBOGAZ ANALIZI ===")
    print(f"  Tek aktor hizi ({num_envs} env)       : {single_actor_throughput:8.1f} env step/sn")
    print(f"  Toplam aktor hizi ({num_actors} aktor)   : {total_actor_throughput:8.1f} env step/sn")
    print(f"  Learner + Reanalyze kapasitesi  : {max_learner_steps_per_sec:8.2f} learner step/sn ({learner_ms:.1f} ms/step)")

    if learner_device.platform == "cpu":
        print("\n  [!] UYARI: Learner CPU uzerinde calisiyor! CUDA etkin bir jaxlib kurulu degil.")
        print("      GPU kullanmak icin: `uv pip install -e '.[cuda12]'`")

    if config.replay_ratio is not None:
        required_env_steps_per_learner_step = config.batch_size / config.replay_ratio
        actor_limited_learner_rate = total_actor_throughput / required_env_steps_per_learner_step
        print(f"\n  Hedef replay_ratio={config.replay_ratio} (1 learner adimi icin {required_env_steps_per_learner_step:.0f} env adimi gerekli):")
        print(f"  - Aktorlerin besleyebildigi hiz : {actor_limited_learner_rate:8.2f} learner step/sn")
        print(f"  - Learner'in isleyebildigi hiz  : {max_learner_steps_per_sec:8.2f} learner step/sn")

        if actor_limited_learner_rate < max_learner_steps_per_sec * 0.85:
            needed_actors = max(1, round(num_actors * (max_learner_steps_per_sec / max(1e-6, actor_limited_learner_rate))))
            print(
                f"\n  -> DARBOGAZ: AKTOR (Veri Uretimi). Learner zamaninin ~%{(1 - actor_limited_learner_rate / max_learner_steps_per_sec) * 100:.0f}'ini"
                f" aktorleri bekleyerek geciriyor.\n"
                f"     Cozum: `--num-actors {needed_actors}` veya `--num-envs` artir, ya da `replay_ratio` degerini yukselt."
            )
        elif max_learner_steps_per_sec < actor_limited_learner_rate * 0.85:
            reanalyze_pct = learner_stages["reanalyze (MCTS + tail)"] / learner_ms * 100.0
            print(
                f"\n  -> DARBOGAZ: LEARNER / REANALYZE. Aktorler zamanin ~%{(1 - max_learner_steps_per_sec / actor_limited_learner_rate) * 100:.0f}'inde"
                f" learner'i bekliyor."
            )
            if reanalyze_pct > 50.0:
                print(
                    f"     En buyuk pay Reanalyze MCTS aramasinda (%{reanalyze_pct:.0f}).\n"
                    f"     Cozum: `reanalyze_ratio` ({config.reanalyze_ratio}) veya `num_simulations` ({config.num_simulations}) dusur."
                )
            else:
                print("     Cozum: Model boyutunu (`--size`) veya `num_unroll_steps` degerini kucult.")
        else:
            print("\n  -> DENGELI: Aktor veri uretim hizi ile Learner isleme hizi birbirine yakin.")


def main() -> None:
    args = tyro.cli(ProfileArgs)
    config = args.agent.with_size(args.size)
    algo = algorithm_for(config)
    actor_device, learner_device = setup_devices()

    env = make_vec_env(args.env, args.num_envs)
    spec = make_environment_spec(env)
    env.close()

    reanalyze_batch = round(config.batch_size * config.reanalyze_ratio)
    reanalyze_searches = reanalyze_batch * min(config.num_unroll_steps + 1, config.sequence_length)
    print(
        f"Algoritma: {algo.name.upper()} (size={args.size}) | Env: {args.env} ({args.num_envs}x{args.num_actors}) | "
        f"Batch: {config.batch_size} | Sims: {config.num_simulations} | "
        f"Reanalyze: {reanalyze_batch}/{config.batch_size} ({reanalyze_searches} MCTS/step)"
    )

    graphdef, params = algo.init(spec, config, jax.random.PRNGKey(0))

    actor_items = profile_actor(args.env, args.num_envs, spec, config, graphdef, params, actor_device, args.actor_iters)
    actor_ms = print_breakdown(f"AKTOR ADIMI ({args.num_envs} env, {actor_device})", actor_items)

    learner_stages, learner_sub = profile_learner(spec, config, graphdef, params, learner_device, args.learner_iters)
    print_breakdown(f"LEARNER + REANALYZE ADIMI (batch={config.batch_size}, {learner_device})", learner_stages)
    print_breakdown("LEARNER GRADIENT_STEP IC DAGILIMI", learner_sub)

    print_diagnosis(config, args.num_envs, args.num_actors, actor_ms, learner_stages, learner_device)


if __name__ == "__main__":
    main()

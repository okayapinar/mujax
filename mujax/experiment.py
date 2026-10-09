"""Experiment setup: wires the pieces together and runs the actor/evaluator threads and the learner."""

from __future__ import annotations

import dataclasses
import functools
import os
import signal
import threading
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import jax
import optax
from tqdm.auto import tqdm

from mujax.actor import Actor
from mujax.algorithm import Algorithm
from mujax.algorithms import ALGORITHMS, algorithm_for
from mujax.checkpoint import BestCheckpointer, CheckpointingConfig, StateCheckpointer, load_params, resolve_checkpoint
from mujax.config import MuZeroConfig
from mujax.learner import Learner
from mujax.loggers import Logger, Writer
from mujax.loop import Counter, EnvHook, EnvironmentLoop
from mujax.observers import ActionFractionObserver, EnvLoopObserver, PolicyEntropyObserver
from mujax.reanalyze import Reanalyzer
from mujax.replay import Buffer
from mujax.types import EnvironmentSpec, make_environment_spec


@dataclasses.dataclass
class EvaluationConfig:
    """How the evaluator score (checkpoint selection and early stopping) is computed."""

    # "episode_return" is the sum of env rewards; any other name is read from the episode result
    # (can also be a key in the env info via InfoKeysObserver).
    metric: str = "episode_return"
    ema_period: int = 14
    # Stops if the score EMA doesn't improve for this many consecutive eval episodes; 0 = disabled.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_min_steps: int = 0


@dataclasses.dataclass
class ExperimentConfig:
    """Definition of a single-process (threaded) experiment.

    Attributes:
        config: Agent hyperparameters.
        max_num_learner_steps / max_num_actor_steps: the run stops when either budget is reached (actor steps are
            summed over the training envs). With `config.replay_ratio` the two are tied:
            learner_steps ~= actor_steps * replay_ratio / batch_size, see `MuZeroConfig.num_learner_steps`.
            Both None: runs until early stopping or Ctrl-C.
        algorithm: if None, chosen from the config class (SMZConfig -> SMZ, GMZConfig -> GMZ).
        environment_factory: seed -> `gym.vector.VectorEnv` for training.
        eval_environment_factory: seed -> `gym.vector.VectorEnv` for evaluation (usually
            num_envs=1); if None, there is no evaluator.
        seed: Seeds the network init, search and replay sampling, and the first env reset (train sub-envs get
            `seed + i`, the evaluator `seed + num_envs + i`).
        observer_factories: Each env loop (actor and evaluator) gets its own instance.
        writer: e.g. `create_writer(...)`; learner, actor and evaluator write to it under `learner/`, `actor/`,
            `evaluator/`. Closed at the end of `run_experiment`. None: only the progress bar.
        hparams: Extra hyperparameters (e.g. env id) written to `writer` together with the algorithm config.
        checkpoint_extra: JSON-serializable info to add to the checkpoint metadata.
        train_env_hook / eval_env_hook: `hook(env, learner_steps)`; see EnvironmentLoop.
    """

    config: MuZeroConfig
    environment_factory: Callable[[int], gym.vector.VectorEnv]
    max_num_learner_steps: int | None = None
    max_num_actor_steps: int | None = None
    algorithm: Algorithm | None = None
    eval_environment_factory: Callable[[int], gym.vector.VectorEnv] | None = None
    seed: int = 0
    observer_factories: Sequence[Callable[[], EnvLoopObserver]] = ()
    writer: Writer | None = None
    hparams: dict | None = None
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
    checkpointing: CheckpointingConfig | None = None
    checkpoint_extra: dict | None = None
    train_env_hook: EnvHook | None = None
    eval_env_hook: EnvHook | None = None


class ScoreTracker:
    """Evaluator episode callback: score EMA, best checkpoint and early stopping.

    Adds `<metric>_ema` to the episode result. The best checkpoint is chosen by EMA, not by a single
    episode score; the saved params are those that played the episode that last updated the EMA.
    """

    def __init__(self, config: EvaluationConfig, actor: Actor, stop_event: threading.Event, checkpointer: BestCheckpointer | None) -> None:
        self._config = config
        self._actor = actor
        self._stop_event = stop_event
        self._checkpointer = checkpointer
        self._alpha = 2.0 / (max(1, int(config.ema_period)) + 1.0)
        self._ema: float | None = None
        self._best_ema = float("-inf")
        self._stale_episodes = 0

    def __call__(self, result: dict[str, float]) -> None:
        metric = self._config.metric
        if metric not in result:
            return
        score = float(result[metric])
        self._ema = score if self._ema is None else self._alpha * score + (1.0 - self._alpha) * self._ema
        result[f"{metric}_ema"] = self._ema

        if self._checkpointer is not None:
            params, step = self._actor.get_params()
            self._checkpointer.maybe_update(self._ema, step, params)
            self._checkpointer.maybe_persist()
        self._check_early_stopping(result.get("learner_steps", 0))

    def _check_early_stopping(self, learner_steps: int) -> None:
        c = self._config
        if c.early_stopping_patience <= 0 or learner_steps < c.early_stopping_min_steps:
            return
        if self._ema > self._best_ema + c.early_stopping_min_delta:
            self._best_ema = self._ema
            self._stale_episodes = 0
            return
        self._stale_episodes += 1
        if self._stale_episodes >= c.early_stopping_patience:
            print(f"Early stopping: eval {c.metric}_ema={self._ema:.6f} iyilesmedi (patience={c.early_stopping_patience})")
            self._stop_event.set()


def setup_devices() -> tuple[jax.Device, jax.Device]:
    """Actor/eval run on CPU, the learner on GPU if available; the default jax device is set to CPU."""

    def first_device(platform: str) -> jax.Device:
        try:
            return jax.devices(platform)[0]
        except RuntimeError:
            return jax.devices()[0]

    actor_device, learner_device = first_device("cpu"), first_device("gpu")
    jax.config.update("jax_default_device", actor_device)
    print(f"Devices -> actor/eval: {actor_device} | learner: {learner_device}")
    return actor_device, learner_device


def make_optimizer(c: MuZeroConfig) -> tuple[optax.GradientTransformation, optax.Schedule]:
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=c.lr_init_value, peak_value=c.learning_rate, warmup_steps=c.lr_warmup_steps, decay_steps=c.lr_decay_steps, end_value=c.lr_end_value
    )
    adamw = optax.adamw(lr_schedule, b1=c.adam_b1, b2=c.adam_b2, weight_decay=c.weight_decay)
    optimizer = optax.chain(optax.clip_by_global_norm(c.max_grad_norm), adamw) if c.max_grad_norm > 0 else adamw
    return optimizer, lr_schedule


class _Worker(threading.Thread):
    """Thread that triggers stop_event on error; the error is re-raised in the main thread."""

    def __init__(self, name: str, target: Callable[[], None], stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self._target_fn = target
        self._stop_event = stop_event
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self._target_fn()
        except BaseException as error:  # noqa: BLE001
            self.error = error
            traceback.print_exc()
            self._stop_event.set()


def run_experiment(experiment: ExperimentConfig) -> Learner:
    """Runs the actor and evaluator in threads and the learner in the main thread; returns the learner."""
    config = experiment.config
    algorithm = experiment.algorithm or algorithm_for(config)
    actor_device, learner_device = setup_devices()
    dataset_key, learner_key, actor_key, eval_key = jax.random.split(jax.random.PRNGKey(experiment.seed), 4)
    stop_event = threading.Event()
    counter = Counter()
    writer = experiment.writer

    def make_logger(label: str) -> Logger | None:
        return None if writer is None else Logger(label, writer)

    environment = experiment.environment_factory(experiment.seed)
    spec = make_environment_spec(environment)
    num_envs = int(environment.num_envs)
    # Train sub-envs use seeds seed..seed+num_envs-1; the evaluator starts right after so no initial state repeats.
    eval_seed = experiment.seed + num_envs
    eval_environment = experiment.eval_environment_factory(eval_seed) if experiment.eval_environment_factory else None
    graphdef, params = algorithm.init(spec, config, learner_key)
    if writer is not None:
        hparams = {"algo": algorithm.name, "seed": experiment.seed, **config.to_dict(), **(experiment.hparams or {})}
        writer.write_config(hparams)

    replay = Buffer(
        spec,
        num_envs=num_envs,
        max_size=config.max_replay_size,
        min_size=config.min_replay_size,
        sample_batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        period=config.replay_period,
        replay_ratio=config.replay_ratio,
        stop_event=stop_event,
    )
    optimizer, lr_schedule = make_optimizer(config)
    learner = Learner(
        functools.partial(algorithm.loss, graphdef, config),
        params,
        optimizer,
        lr_schedule=lr_schedule,
        batch_size=config.batch_size,
        device=learner_device,
        logger=make_logger("learner"),
        counter=counter,
        replay=replay,
    )
    state_checkpointer = None
    if experiment.checkpointing is not None:
        state_checkpointer = StateCheckpointer(experiment.checkpointing, learner)
        if experiment.checkpointing.resume:
            state_checkpointer.restore()
    reanalyze_config = dataclasses.replace(config, num_simulations=config.reanalyze_num_simulations)
    dataset = Reanalyzer(
        replay,
        config=config,
        policy=algorithm.make_policy(graphdef, spec, reanalyze_config, False),
        value_fn=algorithm.make_value_fn(graphdef, config),
        get_params=learner.get_params,
        device=learner_device,
        random_key=dataset_key,
        stop_event=stop_event,
    )

    def make_actor(key: jax.Array, evaluation: bool) -> Actor:
        return Actor(
            algorithm.make_policy(graphdef, spec, config, evaluation),
            key,
            learner.get_params,
            num_actions=spec.num_actions,
            replay=None if evaluation else replay,
            value_fn=None if evaluation else algorithm.make_value_fn(graphdef, config),
            device=actor_device,
            update_period=config.variable_update_period,
            per_episode_update=evaluation,
        )

    def make_observers() -> list[EnvLoopObserver]:
        return [ActionFractionObserver(), PolicyEntropyObserver(), *(factory() for factory in experiment.observer_factories)]

    train_loop = EnvironmentLoop(
        environment,
        make_actor(actor_key, evaluation=False),
        name="actor",
        counter=counter,
        logger=make_logger("actor"),
        observers=make_observers(),
        env_hook=experiment.train_env_hook,
        stop_event=stop_event,
        seed=experiment.seed,
    )
    workers = [_Worker("actor", train_loop.run, stop_event)]

    best_checkpointer = None
    if experiment.checkpointing is not None:
        metadata = {
            "algo": algorithm.name,
            "config": config.to_dict(),
            "environment_spec": spec._asdict(),
            "extra": dict(experiment.checkpoint_extra or {}),
        }
        best_checkpointer = BestCheckpointer(experiment.checkpointing, metadata)
    if eval_environment is not None:
        eval_actor = make_actor(eval_key, evaluation=True)
        eval_loop = EnvironmentLoop(
            eval_environment,
            eval_actor,
            name="evaluator",
            counter=counter,
            logger=make_logger("evaluator"),
            observers=make_observers(),
            episode_callback=ScoreTracker(experiment.evaluation, eval_actor, stop_event, best_checkpointer),
            env_hook=experiment.eval_env_hook,
            stop_event=stop_event,
            seed=eval_seed,
        )
        workers.append(_Worker("evaluator", eval_loop.run, stop_event))

    # Ctrl-C / SIGTERM: the first signal stops gracefully, the second exits immediately.
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():

        def request_stop(*_) -> None:
            if stop_event.is_set():
                os._exit(130)
            print("\nDurdurma sinyali alindi...")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, request_stop)

    def budget_reached() -> bool:
        if experiment.max_num_learner_steps is not None and learner.learn_steps >= experiment.max_num_learner_steps:
            return True
        return experiment.max_num_actor_steps is not None and counter.get().get("actor_steps", 0) >= experiment.max_num_actor_steps

    # Learner loop (main thread)
    for worker in workers:
        worker.start()
    progress = tqdm(
        total=experiment.max_num_learner_steps, initial=learner.learn_steps, desc="Learner", unit="step", dynamic_ncols=True, mininterval=1.0
    )
    try:
        for batch in dataset:
            if budget_reached() or stop_event.is_set():
                break
            learner.step(batch)
            progress.update(1)
            if state_checkpointer is not None:
                state_checkpointer.maybe_save()
    finally:
        stop_event.set()
        progress.close()
        for worker in workers:
            worker.join(timeout=10.0)
        try:
            if best_checkpointer is not None:
                best_checkpointer.persist()
                best_checkpointer.close()
            if state_checkpointer is not None:
                state_checkpointer.save()
                state_checkpointer.close()
        finally:
            environment.close(terminate=True)  # so AsyncVectorEnv subprocesses close without waiting
            if eval_environment is not None:
                eval_environment.close(terminate=True)
            if writer is not None:
                writer.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

    for worker in workers:
        if worker.error is not None:
            raise RuntimeError(f"'{worker.name}' thread'i hata verdi") from worker.error
    return learner


def load_actor(reference: str, *, seed: int = 0, device: jax.Device | None = None) -> tuple[Actor, dict[str, Any]]:
    """Builds an evaluation actor from a checkpoint; for `reference` see `checkpoint.resolve_checkpoint`.

    Returns (actor, metadata); metadata["extra"] is the `checkpoint_extra` given during training.
    """
    step_dir, metadata = resolve_checkpoint(reference)
    algorithm = ALGORITHMS[metadata["algo"]]
    config = algorithm.config_cls.from_dict(metadata["config"])
    spec = EnvironmentSpec(**metadata["environment_spec"])
    graphdef, template = algorithm.init(spec, config, jax.random.PRNGKey(0))
    params = load_params(step_dir, template)
    actor = Actor(
        algorithm.make_policy(graphdef, spec, config, True),
        jax.random.PRNGKey(seed),
        lambda: (params, int(metadata.get("step", 0))),
        num_actions=spec.num_actions,
        device=device,
        per_episode_update=True,
    )
    return actor, metadata

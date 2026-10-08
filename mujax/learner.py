from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from mujax.loggers import Logger
from mujax.loop import Counter
from mujax.replay import Buffer

LossFn = Callable[[Any, Any], tuple[jnp.ndarray, dict[str, jnp.ndarray]]]  # (params, batch) -> (loss, metrics)


class TrainingState(NamedTuple):
    """Training state written to the checkpoint; `step` is the learner step count."""

    params: Any
    opt_state: optax.OptState
    step: int


class Learner:
    """Takes gradient steps with the given loss, logs metrics, and saves/restores its state.

    Algorithm-agnostic: `loss_fn` must be (params, batch) -> (loss, metrics).
    """

    def __init__(
        self,
        loss_fn: LossFn,
        params: Any,
        optimizer: optax.GradientTransformation,
        *,
        lr_schedule: optax.Schedule | None = None,
        batch_size: int = 0,
        device: jax.Device | None = None,
        logger: Logger | None = None,
        counter: Counter | None = None,
        replay: Buffer | None = None,
        log_every: int = 100,
    ):
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule
        self._batch_size = batch_size
        self._logger = logger or Logger("learner")
        self._counter = counter or Counter()
        self._replay = replay  # for metrics only
        self._log_every = max(1, log_every)
        self.learn_steps = 0
        self._restored_steps = 0  # so replay_ratio is matched against this process's actor steps
        self._last_log_time: float | None = None
        self._last_log_steps = 0
        self._params, self._opt_state = jax.device_put((params, optimizer.init(params)), device)
        self._gradient_step = jax.jit(self._gradient_step_fn)

    def get_params(self) -> tuple[Any, int]:
        return self._params, self.learn_steps

    def save(self) -> TrainingState:
        return TrainingState(params=self._params, opt_state=self._opt_state, step=self.learn_steps)

    def restore(self, state: TrainingState) -> None:
        self._params, self._opt_state = state.params, state.opt_state
        self.learn_steps = self._restored_steps = int(state.step)
        # Advance the counter too: env hooks and early stopping read learner_steps from the counter.
        self._counter.increment(learner_steps=self.learn_steps - self._counter.get().get("learner_steps", 0))

    def step(self, batch: Any) -> None:
        self._params, self._opt_state, metrics = self._gradient_step(self._params, self._opt_state, batch)
        self.learn_steps += 1
        counts = self._counter.increment(learner_steps=1)
        if self._last_log_time is None:  # first step (keep compile time out of the rate); also on a resumed run
            self._last_log_time = time.time()
            self._last_log_steps = self.learn_steps - 1
        if self.learn_steps % self._log_every == 0:
            self._log(metrics, counts)

    def _gradient_step_fn(self, params: Any, opt_state: optax.OptState, batch: Any) -> tuple[Any, optax.OptState, dict[str, jnp.ndarray]]:
        (_, metrics), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(params, batch)
        updates, opt_state = self._optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics["grad_norm"] = optax.global_norm(grads)
        metrics["update_norm"] = optax.global_norm(updates)
        metrics["param_norm"] = optax.global_norm(params)
        return params, opt_state, metrics

    def _log(self, metrics: dict[str, jnp.ndarray], counts: dict[str, int]) -> None:
        now = time.time()
        data: dict[str, Any] = {**metrics, "steps_per_second": (self.learn_steps - self._last_log_steps) / max(1e-6, now - self._last_log_time)}
        if self._replay is not None:
            data["buffer_size"] = self._replay.size
            data["buffer_fill_ratio"] = self._replay.fill_ratio
        if counts.get("actor_steps", 0) > 0:
            # Specifically, 2.0 samples were drawn per state, instead of 0.1. appendix H. muzero paper
            data["replay_ratio"] = (self.learn_steps - self._restored_steps) * self._batch_size / counts["actor_steps"]
        if self._lr_schedule is not None:
            data["learning_rate"] = self._lr_schedule(self.learn_steps)
        data.update(counts)
        self._logger.write(data)
        self._last_log_time = now
        self._last_log_steps = self.learn_steps

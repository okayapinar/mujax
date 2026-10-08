"""Optional W&B logging.

`ExperimentConfig.logger_factory` is a `label -> Logger` factory (e.g. `WandbLoggerFactory`); learner, actor and
evaluator each get one logger from it. With no factory nothing is logged (only the tqdm progress bar is shown).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np


class Logger(Protocol):
    def write(self, data: Mapping[str, Any]) -> None: ...


LoggerFactory = Callable[[str], Logger]

# Per-label write intervals to W&B (seconds); other labels use 1 second.
WANDB_EVERY = {"learner": 10.0, "actor": 1.0, "evaluator": 0.0}


def to_scalars(data: Mapping[str, Any]) -> dict[str, float]:
    """Converts scalar numeric values to float; drops everything else (jax arrays are synced here)."""
    result = {}
    for key, value in data.items():
        value = np.asarray(value)
        if value.ndim == 0 and np.issubdtype(value.dtype, np.number):
            result[key] = float(value)
    return result


def import_wandb():
    """wandb is an optional dependency (`pip install mujax[wandb]`); raises a descriptive error if missing."""
    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B icin wandb gerekli: `pip install 'mujax[wandb]'`") from error
    return wandb


class WandbLogger:
    """Logs to the active W&B run as `<label>/<key>` at most once per `every` seconds; the x axis is `step_key`."""

    def __init__(self, label: str, *, every: float = 1.0, step_key: str = "learner_steps") -> None:
        self._wandb = import_wandb()
        self._label = label
        self._every = every
        self._last = float("-inf")
        self._step_key = step_key
        self._wandb.define_metric(step_key)
        self._wandb.define_metric(f"{label}/*", step_metric=step_key)

    def write(self, data: Mapping[str, Any]) -> None:
        now = time.monotonic()
        if now - self._last < self._every:
            return
        self._last = now
        scalars = to_scalars(data)
        payload = {f"{self._label}/{k}": v for k, v in scalars.items() if k != self._step_key}
        payload[self._step_key] = scalars.get(self._step_key, 0.0)
        self._wandb.log(payload)


class WandbLoggerFactory:
    """Opens a W&B run; returns a WandbLogger per label (intervals from `WANDB_EVERY`). `close` finishes the run."""

    def __init__(self, project: str, name: str | None = None, config: dict | None = None, api_key: str | None = None) -> None:
        wandb = import_wandb()
        if api_key:
            wandb.login(key=api_key)
        self._run = wandb.init(project=project, name=name, config=config)

    def __call__(self, label: str) -> Logger:
        return WandbLogger(label, every=WANDB_EVERY.get(label, 1.0))

    def close(self) -> None:
        self._run.finish()

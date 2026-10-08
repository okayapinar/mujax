"""Logger: writes scalars to the terminal and optionally to W&B."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from tqdm.auto import tqdm


def to_scalars(data: Mapping[str, Any]) -> dict[str, float]:
    """Converts scalar numeric values to float; drops everything else (jax arrays are synced here)."""
    result = {}
    for key, value in data.items():
        value = np.asarray(value)
        if value.ndim == 0 and np.issubdtype(value.dtype, np.number):
            result[key] = float(value)
    return result


def _due(every: float | None, last: float, now: float) -> bool:
    return every is not None and (every <= 0 or now - last >= every)


class Logger:
    """Writes to the terminal with a `[label]` prefix and to the active W&B run as `<label>/<key>`.

    `terminal_every` / `wandb_every`: writes to that target at most once per this many seconds; None
    disables the target, 0 lets every write through. In W&B the x axis is the `step_key` counter.
    """

    def __init__(self, label: str, *, terminal_every: float | None = 10.0, wandb_every: float | None = None, step_key: str = "learner_steps") -> None:
        self._label = label
        self._terminal_every = terminal_every
        self._wandb_every = wandb_every
        self._step_key = step_key
        self._last_terminal = 0.0
        self._last_wandb = 0.0
        if wandb_every is not None:
            wandb = import_wandb()
            self._wandb = wandb
            wandb.define_metric(step_key)
            wandb.define_metric(f"{label}/*", step_metric=step_key)

    def write(self, data: Mapping[str, Any]) -> None:
        now = time.time()
        to_terminal = _due(self._terminal_every, self._last_terminal, now)
        to_wandb = _due(self._wandb_every, self._last_wandb, now)
        if not (to_terminal or to_wandb):
            return
        scalars = to_scalars(data)
        if to_terminal:
            self._last_terminal = now
            tqdm.write(f"[{self._label}] " + " | ".join(f"{k}={v:.4g}" for k, v in sorted(scalars.items())))
        if to_wandb:
            self._last_wandb = now
            payload = {f"{self._label}/{k}": v for k, v in scalars.items() if k != self._step_key}
            payload[self._step_key] = scalars.get(self._step_key, 0.0)
            self._wandb.log(payload)


# Per-label write intervals to W&B (seconds).
WANDB_EVERY = {"learner": 10.0, "actor": 1.0, "evaluator": 0.0}


def import_wandb():
    """wandb is an optional dependency (`pip install mujax[wandb]`); raises a descriptive error if missing."""
    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B icin wandb gerekli: `pip install 'mujax[wandb]'`") from error
    return wandb


class WandbLoggerFactory:
    """Opens a W&B run; returns a Logger per label that writes to the terminal + W&B. `close` finishes the run."""

    def __init__(self, project: str, name: str | None = None, config: dict | None = None, api_key: str | None = None) -> None:
        wandb = import_wandb()

        if api_key:
            wandb.login(key=api_key)
        self._run = wandb.init(project=project, name=name, config=config)

    def __call__(self, label: str) -> Logger:
        return Logger(label, wandb_every=WANDB_EVERY.get(label, 1.0))

    def close(self) -> None:
        self._run.finish()

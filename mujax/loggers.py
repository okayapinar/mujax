"""Metric logging to the terminal and Weights & Biases.

    writer = create_writer(console=True, keys=("learner/loss",), wandb_project="mujax")
    ExperimentConfig(..., writer=writer)

Learner, actor and evaluator each get a `Logger`: it writes their metrics as `<label>/<key>` scalars, at most once per
`LOG_EVERY[label]` seconds, with the shared `learner_steps` count as the step. Anything with the `Writer` methods can
be passed as the writer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from tqdm.auto import tqdm

STEP_KEY = "learner_steps"
LOG_EVERY = {"learner": 10.0, "actor": 1.0, "evaluator": 0.0}  # seconds between writes


class Writer(Protocol):
    def write(self, step: int, scalars: Mapping[str, float]) -> None: ...

    def write_config(self, config: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class Logger:
    """Writes one component's metrics (learner, actor or evaluator) to a shared writer."""

    def __init__(self, label: str, writer: Writer) -> None:
        self._label = label
        self._writer = writer
        self._every = LOG_EVERY.get(label, 1.0)
        self._last = float("-inf")

    def write(self, data: Mapping[str, Any]) -> None:
        now = time.monotonic()
        if now - self._last < self._every:
            return
        self._last = now
        scalars = to_scalars(data)
        step = int(scalars.pop(STEP_KEY, 0))
        self._writer.write(step, {f"{self._label}/{key}": value for key, value in scalars.items()})


class ConsoleWriter:
    """Prints to the terminal with `tqdm.write`, so the progress bar stays intact.

    `keys` keeps only those metrics (prefixed names, e.g. `learner/loss`), in that order. None prints all.
    """

    def __init__(self, keys: Sequence[str] | None = None) -> None:
        self._keys = None if keys is None else tuple(keys)

    def write(self, step: int, scalars: Mapping[str, float]) -> None:
        if self._keys is not None:
            scalars = {key: scalars[key] for key in self._keys if key in scalars}
            if not scalars:
                return
        else:
            scalars = dict(sorted(scalars.items()))
        tqdm.write(f"[{step}] " + ", ".join(f"{key}={value:.6g}" for key, value in scalars.items()))

    def write_config(self, config: Mapping[str, Any]) -> None:
        tqdm.write(f"[config] {dict(config)}")

    def close(self) -> None:
        pass


class WandbWriter:
    """Opens a W&B run; `close` finishes it.

    The step is logged as the `learner_steps` metric instead of W&B's own step: learner, actor and evaluator write
    from different threads, so steps can arrive out of order, which W&B's step would reject.
    """

    def __init__(self, project: str, name: str | None = None) -> None:
        self._run = import_wandb().init(project=project, name=name)
        self._run.define_metric(STEP_KEY)
        self._run.define_metric("*", step_metric=STEP_KEY)

    def write(self, step: int, scalars: Mapping[str, float]) -> None:
        self._run.log({**scalars, STEP_KEY: step})

    def write_config(self, config: Mapping[str, Any]) -> None:
        self._run.config.update(dict(config))

    def close(self) -> None:
        self._run.finish()


class MultiWriter:
    """Passes every call to each writer; the lock keeps lines from the actor, evaluator and learner threads apart."""

    def __init__(self, writers: Sequence[Writer]) -> None:
        self._writers = list(writers)
        self._lock = threading.Lock()

    def write(self, step: int, scalars: Mapping[str, float]) -> None:
        with self._lock:
            for writer in self._writers:
                writer.write(step, scalars)

    def write_config(self, config: Mapping[str, Any]) -> None:
        with self._lock:
            for writer in self._writers:
                writer.write_config(config)

    def close(self) -> None:
        with self._lock:
            for writer in self._writers:
                writer.close()


def create_writer(
    *, console: bool = True, wandb_project: str | None = None, wandb_name: str | None = None, keys: Sequence[str] | None = None
) -> MultiWriter:
    """Terminal (if `console`) and W&B (if `wandb_project`) behind one writer.

    `keys` limits the terminal line; W&B still receives every metric.
    """
    writers: list[Writer] = []
    if console:
        writers.append(ConsoleWriter(keys))
    if wandb_project is not None:
        writers.append(WandbWriter(wandb_project, name=wandb_name))
    return MultiWriter(writers)


def to_scalars(data: Mapping[str, Any]) -> dict[str, float]:
    """Keeps numeric scalars as float, drops everything else (jax arrays are synced here)."""
    result = {}
    for key, value in data.items():
        value = np.asarray(value)
        if value.ndim == 0 and np.issubdtype(value.dtype, np.number):
            result[key] = float(value)
    return result


def import_wandb():
    """wandb is optional (`pip install 'mujax[wandb]'`); raises a descriptive error if it's missing."""
    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B icin wandb gerekli: `pip install 'mujax[wandb]'`") from error
    return wandb

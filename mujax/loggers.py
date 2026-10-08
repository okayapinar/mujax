"""Metric logging on CLU `MetricWriter`s.

    writer = create_writer("runs/cartpole", console=True, wandb_project="mujax")   # TensorBoard + terminal + W&B
    ExperimentConfig(..., writer=writer)

Learner, actor and evaluator each get a `Logger`: it writes their metrics as `<label>/<key>` scalars, at most once per
`LOG_EVERY[label]` seconds, with the shared `learner_steps` count as the step.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from clu import metric_writers
from tqdm.auto import tqdm

STEP_KEY = "learner_steps"
LOG_EVERY = {"learner": 10.0, "actor": 1.0, "evaluator": 0.0}  # seconds between writes


class Logger:
    """Writes one component's metrics (learner, actor or evaluator) to a shared writer."""

    def __init__(self, label: str, writer: metric_writers.MetricWriter) -> None:
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
        self._writer.write_scalars(step, {f"{self._label}/{key}": value for key, value in scalars.items()})


class ScalarWriter(metric_writers.MetricWriter):
    """Base for writers that only handle scalars and hparams; the other data types are ignored."""

    def write_summaries(self, step, values, metadata=None):
        pass

    def write_images(self, step, images):
        pass

    def write_videos(self, step, videos):
        pass

    def write_audios(self, step, audios, *, sample_rate):
        pass

    def write_texts(self, step, texts):
        pass

    def write_histograms(self, step, arrays, num_buckets=None):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class ConsoleWriter(ScalarWriter):
    """Prints to the terminal with `tqdm.write`, so the progress bar stays intact."""

    def write_scalars(self, step: int, scalars: Mapping[str, float]) -> None:
        tqdm.write(f"[{step}] " + ", ".join(f"{key}={value:.6g}" for key, value in sorted(scalars.items())))

    def write_hparams(self, hparams: Mapping[str, Any]) -> None:
        tqdm.write(f"[hparams] {dict(hparams)}")


class WandbWriter(ScalarWriter):
    """Opens a W&B run; `close` finishes it.

    The step is logged as the `learner_steps` metric instead of W&B's own step: learner, actor and evaluator write
    from different threads, so steps can arrive out of order, which W&B's step would reject.
    """

    def __init__(self, project: str, name: str | None = None, api_key: str | None = None) -> None:
        wandb = import_wandb()
        if api_key:
            wandb.login(key=api_key)
        self._run = wandb.init(project=project, name=name)
        self._run.define_metric(STEP_KEY)
        self._run.define_metric("*", step_metric=STEP_KEY)

    def write_scalars(self, step: int, scalars: Mapping[str, float]) -> None:
        self._run.log({**scalars, STEP_KEY: step})

    def write_hparams(self, hparams: Mapping[str, Any]) -> None:
        self._run.config.update(dict(hparams))

    def close(self) -> None:
        self._run.finish()


def create_writer(
    log_dir: str | None = None,
    *,
    console: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    wandb_api_key: str | None = None,
) -> metric_writers.MetricWriter:
    """Combines TensorBoard (if `log_dir`), terminal (if `console`) and W&B (if `wandb_project`) into one writer.

    Writes run in background threads, so training doesn't wait on disk or network.
    """
    writers: list[metric_writers.MetricWriter] = []
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        writers.append(metric_writers.SummaryWriter(log_dir))
    if console:
        writers.append(ConsoleWriter())
    if wandb_project is not None:
        writers.append(WandbWriter(wandb_project, name=wandb_name, api_key=wandb_api_key))
    return metric_writers.AsyncMultiWriter(writers)


def to_scalars(data: Mapping[str, Any]) -> dict[str, float]:
    """Keeps numeric scalars as float, drops everything else (jax arrays are synced here)."""
    result = {}
    for key, value in data.items():
        value = np.asarray(value)
        if value.ndim == 0 and np.issubdtype(value.dtype, np.number):
            result[key] = float(value)
    return result


def to_hparams(data: Mapping[str, Any]) -> dict[str, bool | int | float | str]:
    """Turns values TensorBoard's hparams can't take (None, lists, ...) into strings."""
    return {key: value if isinstance(value, (bool, int, float, str)) else str(value) for key, value in data.items()}


def import_wandb():
    """wandb is optional (`pip install 'mujax[wandb]'`); raises a descriptive error if it's missing."""
    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B icin wandb gerekli: `pip install 'mujax[wandb]'`") from error
    return wandb

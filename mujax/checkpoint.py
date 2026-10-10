"""Checkpoints: one directory per training run.

    <directory>/
        best/     params with the best evaluator score; what `load_policy` / `load_actor` read
        latest/   full learner state (params, optimizer, step); what a resumed run continues from

A checkpoint directory holds `pytree/` (Orbax) and `metadata.json`; `save_checkpoint` and `load_pytree` read and
write one. `Checkpointer` drives both directories during training.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import threading
import time
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from mujax.learner import Learner
from mujax.loggers import import_wandb

BEST = "best"
LATEST = "latest"
METADATA_FILE = "metadata.json"
PYTREE_DIR = "pytree"


@dataclasses.dataclass
class CheckpointingConfig:
    directory: str = "checkpoints"
    # `best/` and `latest/` are each written at most once per this many seconds (crash/OOM safeguard); the first
    # write is immediate, the last one happens at the end of the run. 0 = write only at the end of the run.
    every_sec: float = 1800.0
    resume: bool = True  # if `<directory>/latest` exists, the learner continues from it
    upload_to_wandb: bool = True  # if a W&B run is active, every `best/` write is also uploaded as an artifact


# --- Reading and writing a single checkpoint directory -----------------------------------------------------------


def save_checkpoint(directory: str, pytree: Any, metadata: dict[str, Any]) -> None:
    """Writes `<directory>/pytree` and `<directory>/metadata.json`; an existing checkpoint is replaced atomically."""
    directory = os.path.abspath(directory)
    tmp, old = directory + ".tmp", directory + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    os.makedirs(tmp)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(os.path.join(tmp, PYTREE_DIR), pytree)
    with open(os.path.join(tmp, METADATA_FILE), "w") as f:
        json.dump(metadata, f, indent=2)
    if os.path.isdir(directory):
        os.rename(directory, old)
    os.rename(tmp, directory)
    shutil.rmtree(old, ignore_errors=True)


def is_checkpoint(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, METADATA_FILE))


def read_metadata(directory: str) -> dict[str, Any]:
    with open(os.path.join(directory, METADATA_FILE)) as f:
        return json.load(f)


def load_pytree(directory: str, template: Any) -> Any:
    """Reads `<directory>/pytree` into the structure, shapes and dtypes of `template` (a pytree of arrays)."""
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, template)
    with ocp.StandardCheckpointer() as checkpointer:
        return checkpointer.restore(os.path.join(directory, PYTREE_DIR), abstract)


def resolve_checkpoint(reference: str) -> tuple[str, dict[str, Any]]:
    """(directory, metadata) of a best-params checkpoint.

    `reference` is a run directory (`checkpoints/<run>`), a checkpoint directory (`checkpoints/<run>/best`) or a
    W&B artifact reference (e.g. "entity/project/run-checkpoint:best").
    """
    if not os.path.isdir(reference):
        artifact = import_wandb().Api().artifact(reference, type="model")
        reference = os.path.join(artifact.download(), "checkpoint")
    for directory in (reference, os.path.join(reference, BEST)):
        if is_checkpoint(directory):
            return os.path.abspath(directory), read_metadata(directory)
    raise FileNotFoundError(f"Checkpoint bulunamadi: {reference}")


# --- Training-time checkpointer ----------------------------------------------------------------------------------


class _Interval:
    """`due()` is True at most once per `seconds` (the first time immediately); never when `seconds` is 0."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._last: float | None = None

    def due(self) -> bool:
        now = time.monotonic()
        if self._seconds <= 0 or (self._last is not None and now - self._last < self._seconds):
            return False
        self._last = now
        return True


class Checkpointer:
    """Writes `best/` and `latest/` under `config.directory` during training.

    `update_best` is called by the evaluator thread, `save_latest` by the learner thread; each writes at most once
    per `every_sec`. `close` writes whatever is pending. The best score survives a resume: it is read back from
    `best/metadata.json`, so a resumed run only overwrites `best/` with a better score.
    """

    def __init__(self, config: CheckpointingConfig, learner: Learner, metadata: dict[str, Any]) -> None:
        self._config = config
        self._learner = learner
        self._metadata = dict(metadata)
        self.best_dir = os.path.join(config.directory, BEST)
        self.latest_dir = os.path.join(config.directory, LATEST)
        self.best_score = float(read_metadata(self.best_dir)["score"]) if is_checkpoint(self.best_dir) else float("-inf")
        self._pending_best: tuple[int, dict] | None = None  # (step, params as numpy) better than `best/` on disk
        self._best_interval = _Interval(config.every_sec)
        self._latest_interval = _Interval(config.every_sec)
        self._lock = threading.Lock()

    def restore(self) -> bool:
        """Continues the learner from `latest/` if it exists."""
        if not is_checkpoint(self.latest_dir):
            return False
        self._learner.restore(load_pytree(self.latest_dir, self._learner.save()))
        print(f"Egitim durumu geri yuklendi: {self.latest_dir} (step={self._learner.learn_steps})")
        return True

    def update_best(self, score: float, step: int, params: nnx.State) -> None:
        """Remembers `params` if `score` beats the best so far; writes `best/` when the interval allows."""
        with self._lock:
            if float(score) <= self.best_score:
                return
            self.best_score = float(score)
            self._pending_best = (int(step), jax.tree.map(np.asarray, nnx.to_pure_dict(params)))
            if self._best_interval.due():
                self._write_best()

    def save_latest(self, force: bool = False) -> None:
        """Writes `latest/` when the interval allows (or always, with `force`)."""
        if force or self._latest_interval.due():
            save_checkpoint(self.latest_dir, self._learner.save(), {"step": self._learner.learn_steps})

    def close(self) -> None:
        """Writes the pending best params (if any) and the final learner state."""
        with self._lock:
            if self._pending_best is not None:
                self._write_best()
        self.save_latest(force=True)

    def _write_best(self) -> None:
        step, params = self._pending_best
        self._pending_best = None
        metadata = {**self._metadata, "step": step, "score": self.best_score}
        save_checkpoint(self.best_dir, params, metadata)
        print(f"Checkpoint yazildi: {self.best_dir} (step={step}, score={self.best_score:.6f})")
        if self._config.upload_to_wandb:
            _upload_to_wandb(self.best_dir, metadata)


def _upload_to_wandb(directory: str, metadata: dict[str, Any]) -> None:
    """Uploads the checkpoint directory as a `model` artifact of the active W&B run (no-op without one)."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    artifact = wandb.Artifact(name=f"{wandb.run.name}-checkpoint", type="model", metadata=metadata)
    artifact.add_dir(directory, name="checkpoint")
    logged = wandb.run.log_artifact(artifact, aliases=["latest", "best"])
    if not wandb.run.offline:  # offline artifacts are uploaded by `wandb sync`
        logged.wait()

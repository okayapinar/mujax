"""Orbax-based checkpoint layer.

`<directory>/best/<step>/`: best params by evaluator score (+ JSON metadata); read by `resolve_checkpoint` + `load_params`.
`<directory>/state/<step>/`: full learner training state (params, optimizer, step); for resuming.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import ContinuousCheckpointingPolicy

from mujax.learner import Learner
from mujax.loggers import import_wandb

BEST_DIR = "best"
STATE_DIR = "state"
_ITEMS = ("params", "metadata")


@dataclasses.dataclass
class CheckpointingConfig:
    directory: str = "checkpoints"
    every_sec: float = 1800.0  # best params and training state are written to disk at this interval (crash/OOM safeguard); 0 = only at the end
    max_to_keep: int = 1  # number of best-by-score params checkpoints to keep
    resume: bool = True  # if `<directory>/state` holds a training state, the learner resumes from it
    upload_to_wandb: bool = True  # if a W&B run is active, also upload the best checkpoint as an artifact


def _best_manager(directory: str, max_to_keep: int | None = None, create: bool = False, every_sec: float = 0.0) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        best_fn=lambda metrics: metrics["score"],
        best_mode="max",
        create=create,
        save_decision_policy=ContinuousCheckpointingPolicy(minimum_interval_secs=int(every_sec)),
    )
    return ocp.CheckpointManager(os.path.abspath(directory), options=options, item_names=_ITEMS)


class BestCheckpointer:
    """Keeps the best params by evaluator score in memory and periodically persists them.

    Orbax `best_fn` decides which checkpoints to keep; in a resumed run the
    previous bests take part in this comparison too.
    """

    def __init__(self, config: CheckpointingConfig, metadata: dict[str, Any]):
        self._config = config
        self._metadata = dict(metadata)
        self._lock = threading.Lock()
        self.directory = os.path.join(config.directory, BEST_DIR)
        self._manager = _best_manager(self.directory, max_to_keep=config.max_to_keep, create=True, every_sec=config.every_sec)
        best_step = self._manager.best_step()
        self.score = float(self._manager.metrics(best_step)["score"]) if best_step is not None else float("-inf")
        self.step = 0
        self.params: dict | None = None

    def maybe_update(self, score: float, step: int, params: Any) -> None:
        score = float(score)
        with self._lock:
            if score <= self.score:
                return
            self.score = score
            self.step = int(step)
            self.params = jax.tree.map(np.asarray, params._asdict())

    def maybe_persist(self) -> None:
        """Writes if the Orbax save policy allows it (`every_sec` since the last write) and a better checkpoint was found since."""
        if self._config.every_sec > 0 and self.params is not None and self._manager.should_save(self.step):
            self.persist()

    def persist(self) -> None:
        with self._lock:
            if self.params is None or self.step in self._manager.all_steps():
                return
            params, step, score = self.params, self.step, self.score
            metadata = {**self._metadata, "step": step, "score": score}
            self._manager.save(
                step,
                args=ocp.args.Composite(params=ocp.args.StandardSave(params), metadata=ocp.args.JsonSave(metadata)),
                metrics={"score": score},
                force=True,
            )
            self._manager.wait_until_finished()
        print(f"Checkpoint yazildi: {self.directory}/{step} (score={score:.6f})")
        if self._config.upload_to_wandb:
            self._upload_to_wandb(step, metadata)

    def close(self) -> None:
        self._manager.wait_until_finished()
        self._manager.close()

    def _upload_to_wandb(self, step: int, metadata: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:
            return
        step_dir = os.path.join(self.directory, str(step))
        if wandb.run is None or not os.path.isdir(step_dir):  # best_fn may have kept a better one and deleted this
            return
        artifact = wandb.Artifact(name=f"{wandb.run.name}-checkpoint", type="model", metadata=metadata)
        artifact.add_dir(step_dir, name="checkpoint")
        wandb.run.log_artifact(artifact, aliases=["latest", "best"]).wait()


class StateCheckpointer:
    """Periodically writes and restores the learner state.

    Writing is asynchronous: training continues once the device copy finishes, disk writes proceed in the background.
    """

    def __init__(self, config: CheckpointingConfig, learner: Learner):
        self._config = config
        self._learner = learner
        self._manager = ocp.CheckpointManager(
            os.path.abspath(os.path.join(config.directory, STATE_DIR)),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                create=True,
                save_decision_policy=ContinuousCheckpointingPolicy(minimum_interval_secs=int(config.every_sec)),
            ),
        )

    def restore(self) -> None:
        """Restores the latest training state if there is one."""
        step = self._manager.latest_step()
        if step is None:
            return
        template = jax.tree.map(ocp.utils.to_shape_dtype_struct, self._learner.save())
        self._learner.restore(self._manager.restore(step, args=ocp.args.StandardRestore(template)))
        print(f"Egitim durumu geri yuklendi: {self._manager.directory}/{step}")

    def maybe_save(self) -> None:
        """Saves if the Orbax save policy allows it (`every_sec` since the last save and no save in progress)."""
        # Ask the policy first so the device-to-host copy in `learner.save()` only happens when a save is due.
        if self._config.every_sec > 0 and self._manager.should_save(self._learner.learn_steps):
            self.save()

    def save(self) -> None:
        state = self._learner.save()
        if state.step not in self._manager.all_steps():
            self._manager.save(state.step, args=ocp.args.StandardSave(state), force=True)

    def close(self) -> None:
        self._manager.wait_until_finished()
        self._manager.close()


def _resolve_step_dir(reference: str) -> str:
    """Resolves a run directory, `best` directory, single step directory or W&B artifact reference to a step directory."""
    if not os.path.isdir(reference):
        wandb = import_wandb()
        artifact = wandb.Api().artifact(reference, type="model")
        return os.path.join(os.path.abspath(artifact.download()), "checkpoint")
    if os.path.isdir(os.path.join(reference, "params")):
        return reference
    if os.path.isdir(os.path.join(reference, BEST_DIR)):
        reference = os.path.join(reference, BEST_DIR)
    manager = _best_manager(reference)
    step = manager.best_step()
    manager.close()
    if step is None:
        raise FileNotFoundError(f"Checkpoint bulunamadi: {reference}")
    return os.path.join(reference, str(step))


def resolve_checkpoint(reference: str) -> tuple[str, dict[str, Any]]:
    """Finds the (step directory, metadata) pair of the best params checkpoint.

    `reference` is a run directory (`checkpoints/<run>`), its `best` directory, a single step directory,
    or a W&B artifact reference (e.g. "entity/project/run-checkpoint:best").
    """
    step_dir = os.path.abspath(_resolve_step_dir(reference))
    metadata = ocp.Checkpointer(ocp.JsonCheckpointHandler()).restore(os.path.join(step_dir, "metadata"))
    return step_dir, metadata


def load_params(step_dir: str, template: Any) -> Any:
    """Reads the params in a step directory using the shapes/dtypes of `template` (a NamedTuple)."""
    restored = ocp.StandardCheckpointer().restore(os.path.join(step_dir, "params"), template._asdict())
    return type(template)(**restored)

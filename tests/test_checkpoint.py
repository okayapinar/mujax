import numpy as np
from flax import nnx

from mujax.checkpoint import Checkpointer, CheckpointingConfig, load_pytree, read_metadata, resolve_checkpoint, save_checkpoint
from mujax.learner import TrainingState


def Params(w: np.ndarray) -> nnx.State:
    return nnx.State({"w": nnx.Param(w)})


class FakeLearner:
    def __init__(self):
        self.learn_steps = 0
        self.restored = None

    def save(self):
        return TrainingState(params={"w": np.full(3, self.learn_steps, np.float32)}, opt_state={}, step=self.learn_steps)

    def restore(self, state):
        self.restored = state
        self.learn_steps = int(state.step)


def test_save_and_load_replaces_atomically(tmp_path):
    directory = str(tmp_path / "ckpt")
    save_checkpoint(directory, {"w": np.ones(2, np.float32)}, {"score": 1.0})
    save_checkpoint(directory, {"w": np.full(2, 2.0, np.float32)}, {"score": 2.0})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt"]  # no .tmp / .old left behind
    assert read_metadata(directory) == {"score": 2.0}
    np.testing.assert_array_equal(load_pytree(directory, {"w": np.zeros(2, np.float32)})["w"], 2.0)


def test_interval_and_final_writes(tmp_path):
    config = CheckpointingConfig(directory=str(tmp_path), every_sec=3600, upload_to_wandb=False)
    learner = FakeLearner()
    checkpointer = Checkpointer(config, learner, metadata={"algo": "x"})
    for step in range(1, 6):
        learner.learn_steps = step
        checkpointer.save_latest()
        checkpointer.update_best(score=float(step), step=step, params=Params(w=np.full(2, step, np.float32)))
    # The first write is immediate, later ones wait `every_sec`.
    assert read_metadata(checkpointer.latest_dir)["step"] == 1
    assert read_metadata(checkpointer.best_dir)["score"] == 1.0
    checkpointer.close()  # the end of the run writes the pending best and the final state regardless of the interval
    assert read_metadata(checkpointer.latest_dir)["step"] == 5
    assert read_metadata(checkpointer.best_dir) == {"algo": "x", "step": 5, "score": 5.0}

    directory, metadata = resolve_checkpoint(str(tmp_path))
    assert directory == checkpointer.best_dir and metadata["step"] == 5
    assert resolve_checkpoint(checkpointer.best_dir)[0] == directory
    np.testing.assert_array_equal(load_pytree(directory, {"w": np.zeros(2, np.float32)})["w"], 5.0)


def test_only_at_end_and_resume(tmp_path):
    config = CheckpointingConfig(directory=str(tmp_path), every_sec=0, upload_to_wandb=False)
    learner = FakeLearner()
    checkpointer = Checkpointer(config, learner, metadata={})
    assert checkpointer.restore() is False
    learner.learn_steps = 10
    checkpointer.save_latest()
    checkpointer.update_best(score=5.0, step=10, params=Params(w=np.ones(2, np.float32)))
    assert not (tmp_path / "latest").exists() and not (tmp_path / "best").exists()  # every_sec=0: only at the end
    checkpointer.close()

    # A resumed run continues the learner from `latest/` and keeps the previous best score.
    resumed = FakeLearner()
    checkpointer = Checkpointer(config, resumed, metadata={})
    assert checkpointer.restore() is True
    assert resumed.learn_steps == 10
    np.testing.assert_array_equal(resumed.restored.params["w"], 10.0)
    assert checkpointer.best_score == 5.0
    checkpointer.update_best(score=3.0, step=11, params=Params(w=np.zeros(2, np.float32)))
    checkpointer.close()
    assert read_metadata(checkpointer.best_dir)["step"] == 10

import numpy as np
from flax import nnx

from mujax.checkpoint import BestCheckpointer, CheckpointingConfig, StateCheckpointer
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


def test_state_checkpointer_interval_and_restore(tmp_path):
    config = CheckpointingConfig(directory=str(tmp_path), every_sec=3600)
    learner = FakeLearner()
    checkpointer = StateCheckpointer(config, learner)
    for step in range(1, 11):
        learner.learn_steps = step
        checkpointer.maybe_save()
    checkpointer.close()
    # Orbax saves on the first call (no previous checkpoint), then waits `every_sec`.
    assert checkpointer._manager.all_steps() == [1]

    checkpointer = StateCheckpointer(config, learner)
    checkpointer.save()  # final save at the end of training ignores the interval
    checkpointer.close()

    restored = FakeLearner()
    checkpointer = StateCheckpointer(config, restored)
    checkpointer.restore()
    checkpointer.close()
    assert int(restored.restored.step) == 10
    np.testing.assert_array_equal(restored.restored.params["w"], np.full(3, 10, np.float32))


def test_best_checkpointer_interval_and_final_persist(tmp_path):
    config = CheckpointingConfig(directory=str(tmp_path), every_sec=3600, upload_to_wandb=False)
    checkpointer = BestCheckpointer(config, metadata={})
    for step in range(1, 6):
        checkpointer.maybe_update(score=float(step), step=step, params=Params(w=np.full(2, step, np.float32)))
        checkpointer.maybe_persist()
    # Orbax saves on the first improvement (no previous checkpoint), then waits `every_sec`.
    assert checkpointer._manager.all_steps() == [1]
    checkpointer.persist()  # final persist at the end of training ignores the interval; best_fn keeps only the best
    checkpointer.close()
    assert checkpointer._manager.all_steps() == [5]
    assert checkpointer._manager.best_step() == 5

    # A resumed run starts from the previous best score and doesn't write worse params.
    checkpointer = BestCheckpointer(config, metadata={})
    assert checkpointer.score == 5.0
    checkpointer.maybe_update(score=3.0, step=6, params=Params(w=np.zeros(2, np.float32)))
    assert checkpointer.params is None
    checkpointer.close()

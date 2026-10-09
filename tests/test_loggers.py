"""Logger prefixing/throttling and MultiWriter fan-out."""

import numpy as np

from mujax.loggers import ConsoleWriter, Logger, MultiWriter


class _Recorder:
    def __init__(self):
        self.writes, self.configs, self.closed = [], [], False

    def write(self, step, scalars):
        self.writes.append((step, dict(scalars)))

    def write_config(self, config):
        self.configs.append(dict(config))

    def close(self):
        self.closed = True


def test_logger_prefixes_keys_and_uses_learner_steps():
    recorder = _Recorder()
    Logger("evaluator", recorder).write({"episode_return": np.float32(3.0), "learner_steps": 7, "name": "skip", "vec": np.ones(2)})
    assert recorder.writes == [(7, {"evaluator/episode_return": 3.0})]


def test_logger_throttles():
    recorder = _Recorder()
    logger = Logger("learner", recorder)  # at most once per 10 s
    logger.write({"loss": 1.0})
    logger.write({"loss": 2.0})
    assert len(recorder.writes) == 1


def test_console_writer_keeps_only_named_keys(monkeypatch):
    lines = []
    monkeypatch.setattr("mujax.loggers.tqdm.write", lines.append)
    writer = ConsoleWriter(keys=("learner/loss",))
    writer.write(3, {"learner/loss": 0.5, "learner/grad_norm": 1.0})
    writer.write(4, {"actor/episode_return": 10.0})
    assert lines == ["[3] learner/loss=0.5"]


def test_multi_writer_fans_out():
    a, b = _Recorder(), _Recorder()
    writer = MultiWriter([a, b])
    writer.write(1, {"x": 1.0})
    writer.write_config({"seed": 0})
    writer.close()
    for recorder in (a, b):
        assert recorder.writes == [(1, {"x": 1.0})] and recorder.configs == [{"seed": 0}] and recorder.closed

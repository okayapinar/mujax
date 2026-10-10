"""End-to-end smoke tests: a few learner steps, checkpoint and restore."""

import numpy as np
from conftest import make_vec_env

from mujax import (
    CheckpointingConfig,
    EvaluationConfig,
    ExperimentConfig,
    load_actor,
    load_policy,
    run_experiment,
)
from mujax.cli import main
from mujax.config import SIZES


def test_run_and_load(tiny_config, tmp_path):
    experiment = ExperimentConfig(
        config=tiny_config,
        environment_factory=lambda seed: make_vec_env(2),
        eval_environment_factory=lambda seed: make_vec_env(1),
        max_num_learner_steps=5,
        evaluation=EvaluationConfig(),
        checkpointing=CheckpointingConfig(directory=str(tmp_path), every_sec=0, upload_to_wandb=False),
        extra={"env_id": "CartPole-v1"},
    )
    learner = run_experiment(experiment)
    assert learner.learn_steps == 5

    # Learner state is always written at the end; best params only if the evaluator finished an episode.
    assert (tmp_path / "latest").is_dir()
    if not (tmp_path / "best").is_dir():
        return
    policy = load_policy(str(tmp_path))
    assert policy.metadata["extra"] == {"env_id": "CartPole-v1"}
    assert 0 <= policy.act(np.zeros(4, np.float32)) < 2
    assert policy.act(np.zeros((3, 4), np.float32)).shape == (3,)
    actor, metadata = load_actor(str(tmp_path))
    assert metadata["extra"] == {"env_id": "CartPole-v1"}
    env = make_vec_env(1)
    observation, info = env.reset(seed=0)
    actor.observe_first(observation, info)
    action = actor.select_action(observation)
    assert action.shape == (1,) and 0 <= int(action[0]) < 2
    env.close()


def test_single_env_factories(tmp_path):
    import gymnasium as gym
    from conftest import tiny_gmz

    experiment = ExperimentConfig(
        config=tiny_gmz(),
        environment_factory=lambda seed: gym.make("CartPole-v1"),
        eval_environment_factory=lambda seed: gym.make("CartPole-v1"),
        max_num_learner_steps=3,
        checkpointing=CheckpointingConfig(directory=str(tmp_path), every_sec=0, upload_to_wandb=False),
    )
    assert run_experiment(experiment).learn_steps == 3


def test_resume_continues_from_state(tmp_path):
    from conftest import tiny_gmz

    def experiment(steps):
        return ExperimentConfig(
            config=tiny_gmz(),
            environment_factory=lambda seed: make_vec_env(2),
            max_num_learner_steps=steps,
            checkpointing=CheckpointingConfig(directory=str(tmp_path), every_sec=0, upload_to_wandb=False),
        )

    run_experiment(experiment(3))
    assert run_experiment(experiment(6)).learn_steps == 6


def test_cli(tmp_path, monkeypatch):
    monkeypatch.setitem(SIZES, "S", (16, 1, 8))  # preset replaces the layer sizes; keep this smoke test small
    main([
        "--num-envs", "2", "--num-steps", "3", "--size", "S", "--checkpoint-dir", str(tmp_path / "checkpoints"), "--no-console",
        "gmz", "--num-simulations", "4", "--batch-size", "8",
        "--max-replay-size", "2048", "--min-replay-size", "0", "--num-bins", "21",
    ])  # fmt: skip
    assert len(list((tmp_path / "checkpoints").iterdir())) == 1


def test_multiple_actor_threads():
    from conftest import tiny_gmz

    seeds = []

    def factory(seed):
        seeds.append(seed)
        return make_vec_env(2)

    learner = run_experiment(ExperimentConfig(config=tiny_gmz(), environment_factory=factory, num_actors=3, max_num_learner_steps=4))
    assert learner.learn_steps == 4
    assert seeds == [0, 2, 4]  # actor a gets seed + a * num_envs

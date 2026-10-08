"""End-to-end smoke tests: a few learner steps, checkpoint and restore."""

from conftest import make_vec_env

from mujax import (
    CheckpointingConfig,
    EvaluationConfig,
    ExperimentConfig,
    load_actor,
    run_experiment,
)
from mujax.cli import main


def test_run_and_load(tiny_config, tmp_path):
    experiment = ExperimentConfig(
        config=tiny_config,
        environment_factory=lambda seed: make_vec_env(2),
        eval_environment_factory=lambda seed: make_vec_env(1),
        max_num_learner_steps=5,
        evaluation=EvaluationConfig(),
        checkpointing=CheckpointingConfig(directory=str(tmp_path), every_sec=0, upload_to_wandb=False),
        checkpoint_extra={"env_id": "CartPole-v1"},
    )
    learner = run_experiment(experiment)
    assert learner.learn_steps == 5

    # Learner state is always written at the end; best params only if the evaluator finished an episode.
    assert (tmp_path / "state").is_dir()
    if not any((tmp_path / "best").iterdir()):
        return
    actor, metadata = load_actor(str(tmp_path))
    assert metadata["extra"] == {"env_id": "CartPole-v1"}
    env = make_vec_env(1)
    observation, info = env.reset(seed=0)
    actor.observe_first(observation, info)
    action = actor.select_action(observation)
    assert action.shape == (1,) and 0 <= int(action[0]) < 2
    env.close()


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


def test_cli(tmp_path):
    main([
        "--num-envs", "2", "--num-steps", "3", "--no-wandb", "--checkpoint-dir", str(tmp_path),
        "gmz", "--num-simulations", "4", "--reanalyze-num-simulations", "4", "--batch-size", "8",
        "--max-replay-size", "2048", "--num-bins", "21", "--embedding-dim", "8",
        "--representation-layer-sizes", "16", "--prediction-layer-sizes", "16", "--dynamics-layer-sizes", "16",
    ])  # fmt: skip
    assert len(list(tmp_path.iterdir())) == 1

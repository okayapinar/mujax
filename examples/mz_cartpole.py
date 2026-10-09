"""Train MuZero on CartPole.

python examples/mz_cartpole.py
"""

from __future__ import annotations

import gymnasium as gym

from mujax import ConsoleWriter, ExperimentConfig, MZConfig, run_experiment

NUM_STEPS = 20_000


def make_env(num_envs: int) -> gym.vector.VectorEnv:
    # SAME_STEP autoreset is required: the final observation of a truncated episode is read from `info["final_obs"]`.
    return gym.make_vec(
        "CartPole-v1",
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )


def main() -> None:
    experiment = ExperimentConfig(
        config=MZConfig(batch_size=256, reanalyze_ratio=0.0).with_num_steps(NUM_STEPS),
        environment_factory=lambda seed: make_env(16),
        eval_environment_factory=lambda seed: make_env(1),
        max_num_learner_steps=NUM_STEPS,
        writer=ConsoleWriter(keys=("evaluator/episode_return", "evaluator/episode_return_ema", "learner/loss")),
    )
    run_experiment(experiment)


if __name__ == "__main__":
    main()

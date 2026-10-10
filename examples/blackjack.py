"""Train Stochastic MuZero on Blackjack-v1.

Cards are drawn at random. FlattenObservation one-hots the (player sum, dealer card, usable ace) tuple.

python examples/blackjack.py
"""

from __future__ import annotations

import gymnasium as gym

from mujax import ExperimentConfig, SMZConfig, create_writer, run_experiment

NUM_STEPS = 20_000


def make_env(num_envs: int) -> gym.vector.VectorEnv:
    # SAME_STEP autoreset is required: the final observation of a truncated episode is read from `info["final_obs"]`.
    return gym.make_vec(
        "Blackjack-v1",
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
        wrappers=[gym.wrappers.FlattenObservation],
    )


def main() -> None:
    config = SMZConfig(batch_size=128, num_simulations=64, reanalyze_ratio=1, replay_ratio=None).with_size("S")
    experiment = ExperimentConfig(
        config=config.with_num_steps(NUM_STEPS),
        environment_factory=lambda seed: make_env(16),
        eval_environment_factory=lambda seed: make_env(1),
        max_num_learner_steps=NUM_STEPS,
        num_actors=8,
        writer=create_writer(keys=("evaluator/episode_return_ema", "learner/loss")),
    )
    run_experiment(experiment)


if __name__ == "__main__":
    main()

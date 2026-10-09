"""Train Gumbel MuZero (or Stochastic MuZero with `--smz`) on CartPole and keep the best checkpoint.

python examples/cartpole.py
python examples/cartpole.py --smz
"""

from __future__ import annotations

import sys

import gymnasium as gym

from mujax import (
    CheckpointingConfig,
    ExperimentConfig,
    GMZConfig,
    SMZConfig,
    run_experiment,
)

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
    # "S" networks: CartPole does not need more, and the CPU search runs about 4x faster than with the default "M".
    config = (SMZConfig(batch_size=256) if "--smz" in sys.argv else GMZConfig(batch_size=256)).with_size("S")
    experiment = ExperimentConfig(
        config=config.with_num_steps(NUM_STEPS),
        environment_factory=lambda seed: make_env(16),
        eval_environment_factory=lambda seed: make_env(1),
        max_num_learner_steps=NUM_STEPS,
    )
    run_experiment(experiment)


if __name__ == "__main__":
    main()

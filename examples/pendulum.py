"""Train Gumbel MuZero on Pendulum-v1 with three discrete torques.

python examples/pendulum.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from mujax import ExperimentConfig, GMZConfig, create_writer, run_experiment

NUM_STEPS = 20_000


class DiscreteTorque(gym.ActionWrapper):
    """Maps actions 0, 1, 2 onto torques -2, 0, +2."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._torque = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self._torque))

    def action(self, action):
        return np.array([self._torque[int(action)]], dtype=np.float32)


def make_env(num_envs: int) -> gym.vector.VectorEnv:
    # SAME_STEP autoreset is required: the final observation of a truncated episode is read from `info["final_obs"]`.
    return gym.make_vec(
        "Pendulum-v1",
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
        wrappers=[DiscreteTorque],
    )


def main() -> None:
    config = GMZConfig(batch_size=32, reanalyze_ratio=1, replay_ratio=None).with_size("S")
    experiment = ExperimentConfig(
        config=config.with_num_steps(NUM_STEPS),
        environment_factory=lambda seed: make_env(16),
        eval_environment_factory=lambda seed: make_env(1),
        max_num_learner_steps=NUM_STEPS,
        num_actors=8,
        writer=create_writer(keys=("evaluator/episode_return", "learner/loss")),
    )
    run_experiment(experiment)


if __name__ == "__main__":
    main()

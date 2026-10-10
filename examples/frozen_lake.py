"""Train Stochastic MuZero on slippery FrozenLake-v1.

The lake slips, so the same action can move sideways. Reward is 1 only on the goal.

python examples/frozen_lake.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from mujax import ExperimentConfig, SMZConfig, create_writer, run_experiment

NUM_STEPS = 20_000


class OneHotObservation(gym.ObservationWrapper):
    """Discrete state ids are not a distance. One-hot keeps them distinct under symlog."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n = int(env.observation_space.n)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n,), dtype=np.float32)

    def observation(self, observation):
        one_hot = np.zeros(self.observation_space.shape, dtype=np.float32)
        one_hot[int(observation)] = 1.0
        return one_hot


def make_env(num_envs: int) -> gym.vector.VectorEnv:
    # SAME_STEP autoreset is required: the final observation of a truncated episode is read from `info["final_obs"]`.
    return gym.make_vec(
        "FrozenLake-v1",
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
        wrappers=[OneHotObservation],
    )


def main() -> None:
    config = SMZConfig(batch_size=32, num_simulations=16, reanalyze_ratio=1, replay_ratio=None).with_size("S")
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

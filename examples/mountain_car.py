"""Train Gumbel MuZero on MountainCar-v0 and keep the best checkpoint.

python examples/mountain_car.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from mujax import ExperimentConfig, GMZConfig, create_writer, run_experiment

NUM_STEPS = 20_000


def _height(observation: np.ndarray) -> float:
    return float(np.sin(3.0 * observation[0]))


class HeightReward(gym.Wrapper):
    """Adds 100 * (height' - height). Both hills are above the valley, so swinging left pays off."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._height = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._height = _height(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        height = _height(observation)
        reward = float(reward) + 100.0 * (height - self._height)
        self._height = height
        return observation, reward, terminated, truncated, info


def make_env(num_envs: int, *, shaped: bool = False) -> gym.vector.VectorEnv:
    # SAME_STEP autoreset is required: the final observation of a truncated episode is read from `info["final_obs"]`.
    # Shaping sits on the single env, so the bonus uses the real next state, not the autoreset observation.
    return gym.make_vec(
        "MountainCar-v0",
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
        wrappers=[HeightReward] if shaped else None,
    )


def main() -> None:
    config = GMZConfig(batch_size=128, reanalyze_ratio=1, replay_ratio=10.0, num_simulations=32).with_size("S")
    experiment = ExperimentConfig(
        config=config.with_num_steps(NUM_STEPS),
        environment_factory=lambda seed: make_env(16, shaped=True),
        eval_environment_factory=lambda seed: make_env(1),
        max_num_learner_steps=NUM_STEPS,
        num_actors=8,
        writer=create_writer(keys=("evaluator/episode_return", "learner/loss")),
    )
    run_experiment(experiment)


if __name__ == "__main__":
    main()

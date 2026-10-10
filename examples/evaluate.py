"""Play episodes with a trained checkpoint.

python examples/evaluate.py checkpoints/cartpole
python examples/evaluate.py entity/project/run-checkpoint:best   # W&B artifact
"""

from __future__ import annotations

import sys

import gymnasium as gym

from mujax import load_policy


def main(reference: str, num_episodes: int = 5) -> None:
    policy = load_policy(reference)
    env = gym.make(policy.metadata["extra"]["env_id"])
    for episode in range(num_episodes):
        observation, _ = env.reset(seed=episode)
        episode_return, done = 0.0, False
        while not done:
            observation, reward, terminated, truncated, _ = env.step(policy.act(observation))
            episode_return += float(reward)
            done = terminated or truncated
        print(f"episode {episode}: return={episode_return:.1f}")
    env.close()


if __name__ == "__main__":
    main(sys.argv[1])

"""Shared data types: env step, replay transition and env spec."""

from __future__ import annotations

from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np


class EnvironmentSpec(NamedTuple):
    obs_dim: int
    num_actions: int


def make_environment_spec(env: gym.vector.VectorEnv) -> EnvironmentSpec:
    """Derives the spec from the vector env's sub-env spaces (1-D Box observation, Discrete action)."""
    observations, actions = env.single_observation_space, env.single_action_space
    if not isinstance(observations, gym.spaces.Box) or len(observations.shape) != 1:
        raise ValueError(f"1 boyutlu Box gozlem uzayi gerekli, gelen: {observations} (gym.wrappers.FlattenObservation kullan)")
    if not isinstance(actions, gym.spaces.Discrete):
        raise ValueError(f"Discrete aksiyon uzayi gerekli, gelen: {actions}")
    return EnvironmentSpec(obs_dim=int(observations.shape[0]), num_actions=int(actions.n))


class SearchOutput(NamedTuple):
    """Search output; every field has shape (num_envs, ...)."""

    action: np.ndarray
    policy_probs: np.ndarray  # MCTS visit distribution
    value: np.ndarray  # search-improved root value


class TimeStep(NamedTuple):
    """One vector env step; every field has shape (num_envs, ...).

    Under SAME_STEP autoreset, the final info of finished envs is under `info["final_info"]`.
    """

    observation: np.ndarray
    action: np.ndarray
    reward: Any
    terminated: Any
    truncated: Any
    info: dict
    search: SearchOutput | None = None  # search output for this step; read by observers


class Transition(NamedTuple):
    """Replay item. Shaped (num_envs, ...) in the actor and (batch, time, ...) in the learner."""

    observation: Any
    action: Any
    reward: Any
    discount: Any  # 0: episode ended by termination at this step
    policy_probs: Any
    value: Any
    invalid_actions: Any
    truncated: Any  # 1: episode was cut by the time limit at this step (discount stays 1)
    bootstrap_value: Any  # V(final_obs) on a truncated step, otherwise 0

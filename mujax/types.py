"""Shared data types: env step, replay transition and env spec."""

from __future__ import annotations

from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np


class EnvironmentSpec(NamedTuple):
    obs_dim: int
    num_actions: int


def as_vector_env(env: gym.Env | gym.vector.VectorEnv) -> gym.vector.VectorEnv:
    """Returns the env as a SAME_STEP-autoreset vector env.

    A vector env is returned as is (it must already use SAME_STEP autoreset); a single `gym.Env` is wrapped in a
    `SyncVectorEnv` with `num_envs=1`, so the loop, actor and replay only ever see batched `(num_envs, ...)` data.
    Closing the returned env closes the wrapped one.
    """
    if isinstance(env, gym.vector.VectorEnv):
        if env.metadata.get("autoreset_mode") != gym.vector.AutoresetMode.SAME_STEP:
            raise ValueError(
                "Vektor env SAME_STEP autoreset ile kurulmali (NEXT_STEP'teki reset adimi replay'e sahte gecis yazar): "
                'gym.make_vec(..., vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})'
            )
        return env
    if isinstance(env, gym.Env):
        return gym.vector.SyncVectorEnv([lambda: env], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    raise TypeError(f"gym.Env veya gym.vector.VectorEnv bekleniyor, gelen: {type(env).__name__}")


def make_environment_spec(env: gym.Env | gym.vector.VectorEnv) -> EnvironmentSpec:
    """Derives the spec from the env's (sub-env) spaces (1-D Box observation, Discrete action)."""
    if isinstance(env, gym.vector.VectorEnv):
        observations, actions = env.single_observation_space, env.single_action_space
    else:
        observations, actions = env.observation_space, env.action_space
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

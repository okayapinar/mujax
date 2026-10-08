"""EnvironmentLoop observers. Steps are batched as (num_envs, ...)."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import scipy.stats

from mujax.types import TimeStep


class EnvLoopObserver(abc.ABC):
    def observe_first(self, env: gym.vector.VectorEnv, num_envs: int) -> None:
        """Called once at the start of the loop."""

    @abc.abstractmethod
    def observe(self, env: gym.vector.VectorEnv, step: TimeStep) -> None: ...

    @abc.abstractmethod
    def get_metrics(self, index: int) -> dict[str, float]:
        """Called when the episode of sub-env `index` ends; that env's accumulation must be reset."""


class ActionFractionObserver(EnvLoopObserver):
    def observe_first(self, env, num_envs: int) -> None:
        self._counts = np.zeros((num_envs, int(env.single_action_space.n)), dtype=np.int64)

    def observe(self, env, step: TimeStep) -> None:
        actions = np.asarray(step.action, dtype=np.int64)
        self._counts[np.arange(len(actions)), actions] += 1

    def get_metrics(self, index: int) -> dict[str, float]:
        counts = self._counts[index]
        fractions = counts / max(1, int(counts.sum()))
        self._counts[index] = 0
        return {f"action_{i}_fraction": float(p) for i, p in enumerate(fractions)}


class PolicyEntropyObserver(EnvLoopObserver):
    """Mean entropy of the search policy (policy_probs) over an episode."""

    def observe_first(self, env, num_envs: int) -> None:
        self._sum = np.zeros(num_envs, dtype=np.float64)
        self._steps = np.zeros(num_envs, dtype=np.int64)

    def observe(self, env, step: TimeStep) -> None:
        if step.search is None:
            return
        self._sum += scipy.stats.entropy(np.asarray(step.search.policy_probs, dtype=np.float64), axis=-1)
        self._steps += 1

    def get_metrics(self, index: int) -> dict[str, float]:
        if self._steps[index] == 0:
            return {}
        entropy = float(self._sum[index] / self._steps[index])
        self._sum[index] = 0.0
        self._steps[index] = 0
        return {"policy_entropy": entropy}


class InfoKeysObserver(EnvLoopObserver):
    """Reports the given scalar keys from the end-of-episode info (`info["final_info"]` under SAME_STEP)."""

    def __init__(self, keys: Sequence[str]) -> None:
        self._keys = tuple(keys)
        self._final_info: dict | None = None

    def observe(self, env, step: TimeStep) -> None:
        self._final_info = step.info.get("final_info") if step.info else None

    def get_metrics(self, index: int) -> dict[str, float]:
        stats = {}
        for key in self._keys:
            if self._final_info and key in self._final_info:
                value = np.asarray(self._final_info[key])
                stats[key] = float(value if value.ndim == 0 else value[index])
        return stats

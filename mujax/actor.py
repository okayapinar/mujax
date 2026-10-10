from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from mujax.algorithm import SearchPolicy
from mujax.replay import Adder
from mujax.types import SearchOutput, TimeStep, Transition


def invalid_actions_from_info(info: dict | None, num_envs: int, num_actions: int) -> np.ndarray:
    """If `info["invalid_actions"]` (0/1 mask, 1 = invalid) is absent, all actions are treated as valid."""
    raw = info.get("invalid_actions") if info else None
    if raw is None:
        return np.zeros((num_envs, num_actions), dtype=np.float32)
    return np.asarray(raw, dtype=np.float32).reshape(num_envs, num_actions)


class Actor:
    """The env-facing side: jitted search policy + learner params + writing to replay.

    Observations are (num_envs, obs_dim), actions are (num_envs,).

    Args:
        get_params: () -> (params, learner_steps); usually `Learner.get_params`.
        replay: if given (`Buffer.adder(i)`), every step is written there (training actor).
        value_fn: (params, obs) -> search-free value (N,); for episodes cut by the time limit,
            to bootstrap with V(final_obs). If None, a truncated episode is treated as terminated.
        update_period: how many `select_action` calls between params fetches.
        per_episode_update: if True, params are fetched only at episode start (evaluator: the score
            matches the params that played that episode).
    """

    def __init__(
        self,
        policy: SearchPolicy,
        random_key: jax.Array,
        get_params: Callable[[], tuple[Any, int]],
        *,
        num_actions: int,
        replay: Adder | None = None,
        value_fn: Callable[[Any, Any], Any] | None = None,
        device: jax.Device | None = None,
        update_period: int = 1,
        per_episode_update: bool = False,
    ) -> None:
        self._search = jax.jit(policy)
        self._key = random_key
        self._get_params = get_params
        self._num_actions = num_actions
        self._replay = replay
        self._value = jax.jit(value_fn) if value_fn is not None else None
        self._device = device
        self._update_period = max(1, update_period)
        self._per_episode_update = per_episode_update
        self._num_calls = 0
        self._episode_start = True
        self._last_search: SearchOutput | None = None
        self._invalid_actions: np.ndarray | None = None
        self._pull_params()

    def _pull_params(self) -> None:
        params, learner_steps = self._get_params()
        self._params = (jax.device_put(params, self._device), learner_steps)  # single assignment

    def get_params(self) -> tuple[Any, int]:
        """The (params, learner_steps) currently playing."""
        return self._params

    def observe_first(self, observation: np.ndarray, info: dict) -> None:
        self._invalid_actions = invalid_actions_from_info(info, len(observation), self._num_actions)
        self._episode_start = True

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        self._num_calls += 1
        if (self._per_episode_update and self._episode_start) or (not self._per_episode_update and self._num_calls % self._update_period == 0):
            self._pull_params()
        self._episode_start = False

        params, learner_steps = self._params
        self._key, search_key = jax.random.split(self._key)
        search = self._search(params, np.asarray(observation, dtype=np.float32), search_key, self._invalid_actions, learner_steps)
        self._last_search = SearchOutput(
            action=np.asarray(search.action, dtype=np.int32),
            policy_probs=np.asarray(search.policy_probs, dtype=np.float32),
            value=np.asarray(search.value, dtype=np.float32),
        )
        return self._last_search.action

    def _final_values(self, step: TimeStep, truncated: np.ndarray) -> np.ndarray:
        """V(final_obs) for truncated envs; under SAME_STEP the final observation is in `info["final_obs"]`."""
        values = np.zeros(len(truncated), dtype=np.float32)
        if not truncated.any():
            return values
        # To call the jit with a single shape, the whole batch is filled; non-truncated rows are discarded.
        obs = np.zeros_like(np.asarray(step.observation), dtype=np.float32)
        for index in np.flatnonzero(truncated):
            obs[index] = step.info["final_obs"][index]
        values[truncated] = np.asarray(self._value(self._params[0], obs))[truncated]
        return values

    def last_search(self) -> SearchOutput | None:
        """Search output of the last `select_action`; EnvironmentLoop puts it into the TimeStep."""
        return self._last_search

    def observe(self, step: TimeStep) -> None:
        terminated = np.asarray(step.terminated, dtype=bool)
        truncated = np.asarray(step.truncated, dtype=bool) & ~terminated
        done = terminated | truncated
        if self._replay is not None:
            if self._value is None:  # V(final_obs) cannot be computed: a truncated episode is treated as terminated
                terminated, truncated = done, np.zeros_like(truncated)
            self._replay.add(
                Transition(
                    observation=np.asarray(step.observation, dtype=np.float32),
                    action=self._last_search.action,
                    reward=np.asarray(step.reward, dtype=np.float32),
                    discount=np.where(terminated, 0.0, 1.0).astype(np.float32),
                    policy_probs=self._last_search.policy_probs,
                    value=self._last_search.value,
                    invalid_actions=self._invalid_actions,
                    truncated=truncated.astype(np.float32),
                    bootstrap_value=self._final_values(step, truncated),
                )
            )
        self._invalid_actions = invalid_actions_from_info(step.info, len(done), self._num_actions)
        if done.any():
            self._episode_start = True

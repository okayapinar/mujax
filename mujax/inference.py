"""Running a trained policy outside of training: `load_policy` for live inference, `load_actor` for `EnvironmentLoop`."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from flax import nnx

from mujax.actor import Actor
from mujax.algorithm import SearchPolicy
from mujax.algorithms import ALGORITHMS
from mujax.checkpoint import load_pytree, resolve_checkpoint
from mujax.types import EnvironmentSpec, SearchOutput


class Policy:
    """Trained params plus the evaluation search of their algorithm.

        policy = load_policy("checkpoints/cartpole")
        action = policy.act(observation)        # int for one observation, (N,) array for a batch

    `metadata` is the checkpoint metadata; `metadata["extra"]` is `ExperimentConfig.extra` of the training run
    (e.g. the env id). The search is jitted per batch size: the first call with a new batch size compiles.
    """

    def __init__(
        self,
        search: SearchPolicy,
        params: nnx.State,
        num_actions: int,
        *,
        step: int = 0,
        metadata: dict[str, Any] | None = None,
        seed: int = 0,
        device: jax.Device | None = None,
    ) -> None:
        self.params = jax.device_put(params, device)
        self.num_actions = num_actions
        self.step = step
        self.metadata = dict(metadata or {})
        self._search_fn = search
        self._search = jax.jit(search)
        self._key = jax.random.PRNGKey(seed)
        self._device = device

    def search(self, observations: np.ndarray, invalid_actions: np.ndarray | None = None) -> SearchOutput:
        """Search on a batch `(N, obs_dim)`: action, policy_probs and value, all `(N, ...)`.

        `invalid_actions` is an optional 0/1 mask (1 = invalid) of shape `(N, num_actions)`.
        """
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"(N, obs_dim) bekleniyor, gelen sekil: {obs.shape}")
        if invalid_actions is None:
            invalid_actions = np.zeros((len(obs), self.num_actions), dtype=np.float32)
        invalid_actions = np.asarray(invalid_actions, dtype=np.float32).reshape(len(obs), self.num_actions)
        self._key, key = jax.random.split(self._key)
        out = self._search(self.params, obs, key, invalid_actions, self.step)
        return SearchOutput(
            action=np.asarray(out.action, dtype=np.int32),
            policy_probs=np.asarray(out.policy_probs, dtype=np.float32),
            value=np.asarray(out.value, dtype=np.float32),
        )

    def act(self, observation: np.ndarray, invalid_actions: np.ndarray | None = None) -> int | np.ndarray:
        """Action for one observation `(obs_dim,)` (an int) or a batch `(N, obs_dim)` (an `(N,)` array)."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            return int(self.search(obs[None], invalid_actions).action[0])
        return self.search(obs, invalid_actions).action

    def actor(self, *, seed: int = 0) -> Actor:
        """An evaluation `Actor` (batched env interface) with these params, e.g. for `EnvironmentLoop`."""
        return Actor(
            self._search_fn,
            jax.random.PRNGKey(seed),
            lambda: (self.params, self.step),
            num_actions=self.num_actions,
            device=self._device,
            per_episode_update=True,
        )


def load_policy(reference: str, *, seed: int = 0, device: jax.Device | None = None) -> Policy:
    """Loads the best params of a run; for `reference` see `checkpoint.resolve_checkpoint`."""
    directory, metadata = resolve_checkpoint(reference)
    algorithm = ALGORITHMS[metadata["algo"]]
    config = algorithm.config_cls.from_dict(metadata["config"])
    spec = EnvironmentSpec(**metadata["environment_spec"])
    graphdef, params = algorithm.init(spec, config, jax.random.PRNGKey(0))
    nnx.replace_by_pure_dict(params, load_pytree(directory, nnx.to_pure_dict(params)))
    search = algorithm.make_policy(graphdef, spec, config, True)
    return Policy(search, params, spec.num_actions, step=metadata["step"], metadata=metadata, seed=seed, device=device)


def load_actor(reference: str, *, seed: int = 0, device: jax.Device | None = None) -> tuple[Actor, dict[str, Any]]:
    """`load_policy(...).actor()` together with the checkpoint metadata."""
    policy = load_policy(reference, device=device)
    return policy.actor(seed=seed), policy.metadata

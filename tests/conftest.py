"""Small configs and envs used in tests."""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import pytest

from mujax.algorithms import GMZConfig, MZConfig, SampledMZConfig, SMZConfig

# Networks and search small enough to compile and run within seconds.
TINY = {
    "num_simulations": 4,
    "reanalyze_num_simulations": 4,
    "num_unroll_steps": 2,
    "num_bootstrapping": 2,
    "num_bins": 21,
    "batch_size": 8,
    "max_replay_size": 2048,
    "min_replay_size": 0,
    "lr_warmup_steps": 1,
    "lr_decay_steps": 10,
    "representation_layer_sizes": (16,),
    "prediction_layer_sizes": (16,),
    "embedding_dim": 8,
}


def tiny_smz() -> SMZConfig:
    return dataclasses.replace(
        SMZConfig(), **TINY, encoder_layer_sizes=(16,), decision_layer_sizes=(16,), chance_layer_sizes=(16,), codebook_size=4
    )


def tiny_mz() -> MZConfig:
    return dataclasses.replace(MZConfig(), **TINY, dynamics_layer_sizes=(16,))


def tiny_gmz() -> GMZConfig:
    return dataclasses.replace(GMZConfig(), **TINY, dynamics_layer_sizes=(16,), max_num_considered_actions=2)


def tiny_sampled_mz() -> SampledMZConfig:
    return dataclasses.replace(SampledMZConfig(), **TINY, dynamics_layer_sizes=(16,), num_sampled_actions=3)


def make_vec_env(num_envs: int = 2) -> gym.vector.VectorEnv:
    return gym.make_vec(
        "CartPole-v1", num_envs=num_envs, vectorization_mode="sync", vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    )


@pytest.fixture(params=["mz", "smz", "gmz", "sampled_mz"])
def tiny_config(request):
    return {"mz": tiny_mz, "smz": tiny_smz, "gmz": tiny_gmz, "sampled_mz": tiny_sampled_mz}[request.param]()

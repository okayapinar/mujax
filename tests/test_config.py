import dataclasses

import pytest

from mujax import GMZ, SMZ, GMZConfig, MuZeroConfig, SMZConfig, algorithm_for


@pytest.mark.parametrize("cls", [MuZeroConfig, SMZConfig, GMZConfig])
def test_dict_round_trip(cls):
    config = cls()
    assert config.discount == 1.0 - 1.0 / config.effective_horizon
    assert cls.from_dict(config.to_dict()) == config


def test_from_dict_ignores_unknown_fields():
    data = {**SMZConfig().to_dict(), "removed_field": 1}
    assert SMZConfig.from_dict(data) == SMZConfig()


def test_with_num_steps():
    config = SMZConfig().with_num_steps(1000)
    assert config.lr_warmup_steps == 100
    assert config.temperature_decay_steps == 1000
    assert SMZConfig(lr_warmup_steps=50).with_num_steps(1000).lr_warmup_steps == 50


def test_algorithm_for():
    assert algorithm_for(SMZConfig()) is SMZ
    assert algorithm_for(GMZConfig()) is GMZ
    with pytest.raises(ValueError):
        algorithm_for(MuZeroConfig())


@pytest.mark.parametrize("cls", [SMZConfig, GMZConfig])
def test_with_size(cls):
    config = cls().with_size("l")
    assert config.embedding_dim == 128
    for field in dataclasses.fields(config):
        if field.name.endswith("_layer_sizes"):
            assert getattr(config, field.name) == (512, 512, 512), field.name
    with pytest.raises(ValueError):
        cls().with_size("huge")


def test_size_m_is_the_default():
    assert GMZConfig().with_size("M") == GMZConfig()


def test_num_learner_steps():
    config = GMZConfig(batch_size=256, replay_ratio=2.0)
    assert config.num_learner_steps(1_000_000) == 1_000_000 * 2 // 256
    with pytest.raises(ValueError):
        dataclasses.replace(config, replay_ratio=None).num_learner_steps(1000)

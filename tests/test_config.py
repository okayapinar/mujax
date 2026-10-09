import pytest

from mujax import GMZ, MZ, SMZ, GMZConfig, MuZeroConfig, MZConfig, SMZConfig, algorithm_for


@pytest.mark.parametrize("cls", [MuZeroConfig, MZConfig, SMZConfig, GMZConfig])
def test_dict_round_trip(cls):
    config = cls()
    assert config.discount == 1.0 - 1.0 / config.effective_horizon
    assert cls.from_dict(config.to_dict()) == config


def test_from_dict_ignores_unknown_fields():
    data = {**SMZConfig().to_dict(), "removed_field": 1}
    assert SMZConfig.from_dict(data) == SMZConfig()


@pytest.mark.parametrize("cls", [MZConfig, SMZConfig])
def test_with_num_steps(cls):
    config = cls().with_num_steps(1000)
    assert config.lr_warmup_steps == 100
    assert config.lr_warmup_steps + config.lr_decay_steps == 1000
    assert config.temperature_decay_steps == 1000


def test_algorithm_for():
    assert algorithm_for(MZConfig()) is MZ
    assert algorithm_for(SMZConfig()) is SMZ
    assert algorithm_for(GMZConfig()) is GMZ
    with pytest.raises(ValueError):
        algorithm_for(MuZeroConfig())

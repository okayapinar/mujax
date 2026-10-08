"""Hyperparameters shared by SMZ and GMZ."""

from __future__ import annotations

import dataclasses
from typing import Any, Self


@dataclasses.dataclass
class MuZeroConfig:
    """Hyperparameters shared by SMZ and GMZ."""

    # Rollout / targets
    discount: float = 0.997
    num_unroll_steps: int = 5
    num_bootstrapping: int = 10  # appendix H. muzero paper
    bootstrapping_lambda: float = 0.5

    # Search
    num_simulations: int = 32
    reanalyze_num_simulations: int = 64

    # Categorical value/reward support (in symlog space): +-10 corresponds to roughly +-22000, bin width 0.1.
    support_min: float = -10.0
    support_max: float = 10.0
    num_bins: int = 201
    hl_gauss_sigma_scale: float = 0.75
    latent_gradient_scale: float = 0.5

    # Replay / reanalyze
    max_replay_size: int | None = None  # If None, computed from free RAM
    replay_period: int = 1
    reanalyze_ratio: float = 0.5

    # Optimization
    batch_size: int = 1024
    learning_rate: float = 3e-4
    lr_init_value: float = 0.0
    lr_end_value: float = 1e-5
    lr_warmup_steps: int = 5000
    lr_decay_steps: int = 395_000  # optax warmup_cosine_decay_schedule decay_steps
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0
    value_loss_weight: float = 1.0

    # Networks
    representation_layer_sizes: tuple[int, ...] = (256, 256, 256)
    prediction_layer_sizes: tuple[int, ...] = (256, 256, 256)
    embedding_dim: int = 64

    # How many select_action calls between actor pulls of the learner params
    variable_update_period: int = 1

    @property
    def sequence_length(self) -> int:
        return self.num_unroll_steps + self.num_bootstrapping + 1

    def with_num_steps(self, num_steps: int) -> Self:
        """Adjusts the schedules (LR warmup/cosine) to the total number of learner steps."""
        warmup = min(self.lr_warmup_steps, num_steps // 10)
        return dataclasses.replace(self, lr_warmup_steps=warmup, lr_decay_steps=num_steps - warmup)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (tuples become lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Inverse of `to_dict`; fields that no longer exist (old checkpoints) are skipped."""
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in fields})

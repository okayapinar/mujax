"""Hyperparameters shared by SMZ and GMZ."""

from __future__ import annotations

import dataclasses
from typing import Any, Self

# Network size presets, `MuZeroConfig.with_size`: name -> (width, depth, embedding_dim). Every `*_layer_sizes` field
# becomes (width,) * depth. "M" is the default of MZ / GMZ / SampledMZ.
SIZES: dict[str, tuple[int, int, int]] = {
    "S": (128, 2, 32),
    "M": (256, 3, 64),
    "L": (512, 3, 128),
    "XL": (1024, 4, 256),
}


@dataclasses.dataclass
class MuZeroConfig:
    """Hyperparameters shared by SMZ and GMZ."""

    # Rollout / targets
    effective_horizon: int = 333  # discount = 1 - 1/H
    num_unroll_steps: int = 5
    num_bootstrapping: int = 10  # appendix H. muzero paper
    bootstrapping_lambda: float = 0.95

    # Search
    num_simulations: int = 32
    reanalyze_num_simulations: int = 64

    # Categorical value/reward support (in symlog space): +-10 corresponds to roughly +-22000, bin width 0.1.
    support_min: float = -10.0
    support_max: float = 10.0
    num_bins: int = 255
    hl_gauss_sigma_scale: float = 0.75
    latent_gradient_scale: float = 0.5

    # Replay / reanalyze
    max_replay_size: int | None = 5_000_000  # None: computed from free RAM
    min_replay_size: int = 10_000  # Transitions (all envs) in replay before the learner starts; capped at the buffer size
    replay_period: int = 1
    reanalyze_ratio: float = 0.5
    replay_ratio: float | None = 2.0

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
    def discount(self) -> float:
        return 1.0 - 1.0 / self.effective_horizon

    @property
    def sequence_length(self) -> int:
        return self.num_unroll_steps + self.num_bootstrapping + 1

    def with_num_steps(self, num_steps: int) -> Self:
        """Adjusts the schedules (LR warmup/cosine) to the total number of learner steps."""
        warmup = min(self.lr_warmup_steps, num_steps // 10)
        return dataclasses.replace(self, lr_warmup_steps=warmup, lr_decay_steps=num_steps - warmup)

    def with_size(self, size: str) -> Self:
        """Sets every `*_layer_sizes` field and `embedding_dim` from the `SIZES` preset ("S", "M", "L", "XL")."""
        try:
            width, depth, embedding_dim = SIZES[size.upper()]
        except KeyError:
            raise ValueError(f"Bilinmeyen boyut {size!r}; secenekler: {', '.join(SIZES)}") from None
        layers = {f.name: (width,) * depth for f in dataclasses.fields(self) if f.name.endswith("_layer_sizes")}
        return dataclasses.replace(self, embedding_dim=embedding_dim, **layers)

    def num_learner_steps(self, num_actor_steps: int) -> int:
        """Learner steps that `replay_ratio` allows for a budget of `num_actor_steps` environment steps."""
        if self.replay_ratio is None:
            raise ValueError("replay_ratio None iken aktor adimi butcesi learner adimina cevrilemez")
        return max(1, int(num_actor_steps * self.replay_ratio / self.batch_size))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (tuples become lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Inverse of `to_dict`; fields that no longer exist (old checkpoints) are skipped."""
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in fields})

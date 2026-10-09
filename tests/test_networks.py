import jax
import jax.numpy as jnp
import numpy as np
import rlax
from flax import nnx

from mujax.algorithms import MZConfig, SMZConfig
from mujax.algorithms.mz import MZModel
from mujax.algorithms.muzero import Embedding, SymlogInput, unimix_logits
from mujax.algorithms.smz import SMZModel
from mujax.types import EnvironmentSpec


def test_symlog_input_wraps_net():
    net = Embedding(3, (8,), 4, rngs=nnx.Rngs(0))
    obs = jnp.array([[1e4, -2.0, 0.5], [0.0, 7.0, -300.0]])
    np.testing.assert_allclose(SymlogInput(net)(obs), net(rlax.signed_logp1(obs)), rtol=1e-6)


def test_models_read_observations_through_symlog():
    spec = EnvironmentSpec(obs_dim=3, num_actions=2)
    obs = jnp.array([[1e4, -2.0, 0.5]])
    mz = MZModel(spec, MZConfig(representation_layer_sizes=(8,), embedding_dim=4, num_bins=5), rngs=nnx.Rngs(0))
    assert isinstance(mz.representation, SymlogInput)
    np.testing.assert_allclose(mz.representation(obs), mz.representation.net(rlax.signed_logp1(obs)), rtol=1e-6)

    smz = SMZModel(spec, SMZConfig(representation_layer_sizes=(8,), encoder_layer_sizes=(8,), embedding_dim=4, num_bins=5, codebook_size=3), rngs=nnx.Rngs(0))
    assert isinstance(smz.representation, SymlogInput) and isinstance(smz.encoder, SymlogInput)
    np.testing.assert_allclose(smz.encoder(obs), smz.encoder.net(rlax.signed_logp1(obs)), rtol=1e-6)
    assert jnp.all(jnp.isfinite(jax.grad(lambda o: smz.representation(o).sum())(obs)))


def test_unimix_keeps_a_probability_floor():
    logits = jnp.array([[50.0, 0.0, -50.0, -50.0]])
    probs = jax.nn.softmax(unimix_logits(logits, 0.01))
    np.testing.assert_allclose(probs.sum(-1), 1.0, rtol=1e-6)
    assert jnp.all(probs >= 0.01 / 4 - 1e-7)
    np.testing.assert_allclose(probs[0, 0], 0.99 + 0.01 / 4, rtol=1e-5)
    np.testing.assert_array_equal(unimix_logits(logits, 0.0), logits)


def test_models_apply_unimix_to_policy_logits():
    spec = EnvironmentSpec(obs_dim=3, num_actions=4)
    obs = jnp.array([[1e4, -2.0, 0.5]])
    for unimix in (0.0, 0.05):
        model = MZModel(spec, MZConfig(representation_layer_sizes=(8,), embedding_dim=4, num_bins=5, policy_unimix=unimix), rngs=nnx.Rngs(0))
        latent = model.representation(obs)
        _, policy_logits = model.prediction(latent)
        np.testing.assert_allclose(policy_logits, unimix_logits(model.prediction.policy(latent), unimix), rtol=1e-6)
        assert jnp.all(jax.nn.softmax(policy_logits) >= unimix / 4 - 1e-7)

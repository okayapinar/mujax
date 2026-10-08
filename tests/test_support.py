import jax.numpy as jnp
import numpy as np

from mujax.algorithms.muzero import Support


def test_probs_are_normalized_and_invert():
    support = Support(-10.0, 10.0, 201)
    x = jnp.array([-500.0, -3.0, 0.0, 0.5, 42.0, 1000.0])
    probs = support.to_probs(x)
    assert probs.shape == (6, 201)
    np.testing.assert_allclose(probs.sum(-1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(support.to_scalar(probs), x, rtol=1e-2, atol=1e-2)

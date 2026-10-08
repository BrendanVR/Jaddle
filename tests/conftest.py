"""Shared test setup.

Tests run on CPU in double precision by default: small problems are faster on
CPU than on a GPU (no transfer or launch overhead), results are deterministic,
and CI runners have no GPU. Set JAX_PLATFORMS before running pytest to override.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jaddle.jaddle_optimisers as jo  # noqa: E402

jo.configure_jax("float64")

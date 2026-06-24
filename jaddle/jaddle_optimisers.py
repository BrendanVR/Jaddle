import os

import optax
import jax
import jax.numpy as jnp
from typing import Any, Optional
from jaddle.jaddle_basic_types import (
    ScheduleLike,
    LP,
    SaddleState,
)
import os

# The three supported precision profiles. Aliases map legacy names onto them.
_PROFILE_ALIASES = {
    "x64": "float64",
    "safe": "float64",
    "balanced": "float32",
    "max_speed": "float32",
    "float64": "float64",
    "float32": "float32",
    "float16": "float16",
}

# Default JAX array dtype implied by each profile. Read this from solver code
# (e.g. via jaddle_dtype()) to allocate state in the active precision.
_PROFILE_DTYPE = {
    "float64": "float64",
    "float32": "float32",
    "float16": "float16",
}


def jaddle_dtype():
    """Return the JAX dtype for the currently-configured precision profile."""
    profile = _PROFILE_ALIASES.get(
        os.environ.get("JADDLE_JAX_PROFILE", "float64").lower(), "float64"
    )
    return getattr(jnp, _PROFILE_DTYPE[profile])


def configure_jax(jax_profile: Optional[str] = None):
    """
    Configure JAX for one of three precision/performance profiles.

    Args:
        jax_profile: One of:
            - "float64": double precision, maximum safety and accuracy. The
              analogue of PDLP's float64 default. On wide-dynamic-range MIPLIB
              rows, float32 residuals plateau on rounding and "stalled" becomes
              indistinguishable from "out of precision". Full-precision matmuls
              (no TF32) so the extra mantissa actually counts.
            - "float32": single precision, maximum speed. TF32 matmuls.
            - "float16": half precision, maximum speed. TF32 matmuls; the
              tightest memory/bandwidth footprint, at the cost of precision.
            Legacy aliases ("safe"/"x64", "balanced", "max_speed") are accepted.
            If None, reads JADDLE_JAX_PROFILE, defaulting to "float64".

    Must run before JAX initialises for the env vars to take full effect; the
    live jax.config is also updated so it works after import. Prints the active
    configuration for verification.
    """
    if jax_profile is not None:
        os.environ["JADDLE_JAX_PROFILE"] = jax_profile

    raw = os.environ.get("JADDLE_JAX_PROFILE", "float64").lower()
    PROFILE = _PROFILE_ALIASES.get(raw, "float64")
    os.environ["JADDLE_JAX_PROFILE"] = PROFILE

    def _append_xla_flag(flag: str):
        current = os.environ.get("XLA_FLAGS", "")
        if flag not in current:
            os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()

    # Suppress INFO and WARNING logs from XLA/JAX
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/jaddle_jax"),
    )

    if PROFILE == "float64":
        # Double precision: enable x64 and use full-precision (highest) matmuls.
        os.environ["JAX_ENABLE_X64"] = "1"
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
        # The env var only takes effect if read before JAX initialises; since
        # this module already imported jax, set the live config too so x64
        # works even when configure_jax is called after import.
        jax.config.update("jax_enable_x64", True)
        dtype = "float64"
    else:
        # float32 / float16: maximum speed. No x64; aggressive GPU autotuning
        # and memory preallocation. (JAX has no global "default to 16-bit" mode
        # — JAX_ENABLE_X64 only switches float32<->float64 — so half precision
        # is carried by the dtype that jaddle_dtype() hands to state allocation,
        # not by a config flag.)
        os.environ["JAX_ENABLE_X64"] = "0"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
        _append_xla_flag("--xla_gpu_autotune_level=4")
        jax.config.update("jax_enable_x64", False)
        dtype = _PROFILE_DTYPE[PROFILE]

        if PROFILE == "float16":
            # Half precision: accumulate matmuls in float32 so f16 inputs don't
            # lose the running sum to rounding, and keep softmax/reduction math
            # in f32 for stability. The f16 footprint comes from the array dtype;
            # this just makes the reductions over it survivable.
            os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "float32")
            jax.config.update("jax_default_matmul_precision", "float32")
        else:
            # float32: TF32 matmuls trade a few mantissa bits for throughput.
            os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "tensorfloat32")

    print(
        "[JAX Profile] "
        f"mode={PROFILE}, "
        f"dtype={dtype}, "
        f"x64={os.environ.get('JAX_ENABLE_X64', 'default')}, "
        f"matmul_precision={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'default')}, "
        f"cache={os.environ.get('JAX_COMPILATION_CACHE_DIR', 'disabled')}, "
        f"xla_flags='{os.environ.get('XLA_FLAGS', '')}'"
    )


def _saddle_param_labels(params):
    if hasattr(params, "_fields"):
        return type(params)(
            primal="primal_opt",
            dual_ineq="dual_opt",
            dual_eq="dual_opt",
        )
    return {
        "primal": "primal_opt",
        "dual_ineq": "dual_opt",
        "dual_eq": "dual_opt",
    }


def create_saddle_optimiser(
    primal_optimizer: optax.GradientTransformation,
    dual_optimizer: optax.GradientTransformation,
):
    if dual_optimizer is None:
        dual_optimizer = primal_optimizer
    optimiser = optax.partition(
        {
            "primal_opt": primal_optimizer,
            "dual_opt": dual_optimizer,
        },
        param_labels=_saddle_param_labels,
    )

    return optimiser


def optimisitic_gd(lr):
    primal = optax.inject_hyperparams(optax.optimistic_gradient_descent)(
        learning_rate=lr,
    )
    dual = optax.inject_hyperparams(optax.optimistic_gradient_descent)(
        learning_rate=lr,
    )
    return create_saddle_optimiser(
        primal,
        dual,
    )


def gd_dual_momentum(lr, momentum=0.3, nesterov=True):
    primal = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
    )
    dual = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
        momentum=momentum,
        nesterov=nesterov,
    )
    return create_saddle_optimiser(
        primal,
        dual,
    )


def tail_average(i_max):
    """Returns a function that computes the tail average of a sequence."""

    def weight_fn(i):
        return jax.lax.cond(
            i < i_max,
            lambda: 1 / i_max,
            lambda: 1.0,
        )

    return weight_fn

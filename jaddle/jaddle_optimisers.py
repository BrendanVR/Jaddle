import os

import optax
import jax
import jax.numpy as jnp
from typing import Any, Optional
from jaddle.jaddle_basic_types import (
    ExtragradientState,
    ScheduleLike,
    LP,
    SaddleState,
)
import os


def configure_jax(jax_profile: Optional[str] = None):
    """
    Configure JAX environment variables for different profiling modes.

    Args:
        jax_profile: Optional string to specify the profiling mode. Can be "balanced", "max_speed", or None.
                     If None, it will read from the JADDLE_JAX_PROFILE environment variable, defaulting to "max_speed".

    This function sets environment variables to optimize JAX's performance based on the chosen profile.
    It also prints out the active configuration for verification.
    """
    if jax_profile is not None:
        os.environ["JADDLE_JAX_PROFILE"] = jax_profile

    JAX_PROFILE = os.environ.get("JADDLE_JAX_PROFILE", "safe").lower()

    def _append_xla_flag(flag: str):
        current = os.environ.get("XLA_FLAGS", "")
        if flag not in current:
            os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()

    # Suppress INFO and WARNING logs from XLA/JAX
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if JAX_PROFILE in ["balanced", "max_speed"]:
        os.environ.setdefault("JAX_ENABLE_X64", "0")
        os.environ.setdefault(
            "JAX_COMPILATION_CACHE_DIR",
            os.path.expanduser("~/.cache/jaddle_jax"),
        )

    if JAX_PROFILE == "max_speed":
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "tensorfloat32")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
        _append_xla_flag("--xla_gpu_autotune_level=4")
    elif JAX_PROFILE in ["safe", "balanced"]:
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "high")

    # The "x64" profile enables double precision for the convergence-critical
    # path — the analogue of PDLP's float64 default. On wide-dynamic-range
    # MIPLIB rows, float32 residuals plateau on rounding and "stalled" becomes
    # indistinguishable from "out of precision". Must run before JAX initialises.
    # Use full-precision matmuls (no TF32) so the extra mantissa actually counts.
    if JAX_PROFILE == "x64":
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
        os.environ.setdefault(
            "JAX_COMPILATION_CACHE_DIR",
            os.path.expanduser("~/.cache/jaddle_jax"),
        )
        # The env var only takes effect if read before JAX initialises; since
        # this module already imported jax, set the live config too so x64 works
        # even when configure_jax("x64") is called after import.
        jax.config.update("jax_enable_x64", True)

    print(
        "[JAX Profile] "
        f"mode={JAX_PROFILE}, "
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

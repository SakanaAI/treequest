from treequest.imports import try_import

with try_import() as _import:
    import jax  # type: ignore[import-not-found]
    from packaging.version import Version

    # TODO: Remove this hotfix after numpyro fixes incompatibility with jax>=0.7
    # https://github.com/pyro-ppl/numpyro/issues/2051
    if Version(jax.__version__) >= Version("0.7.0"):
        import jax.experimental.pjit as _pjit  # type: ignore[import-not-found]
        from jax.extend.core.primitives import jit_p  # type: ignore[import-not-found]

        _pjit.pjit_p = jit_p  # type: ignore
    import numpy as np
    import numpyro  # type: ignore
    import pandas as pd  # type: ignore
    import pymc as pm  # type: ignore
    from pymc.sampling.jax import sample_numpyro_nuts  # type: ignore
    from xarray import DataArray  # type: ignore[import-not-found]

__all__ = [
    "jax",
    "np",
    "numpyro",
    "pd",
    "pm",
    "sample_numpyro_nuts",
    "DataArray",
]

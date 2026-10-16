import importlib
import sys
from pathlib import Path

from treequest.imports import try_import

_ARVIZ_IMPORT_ATTEMPTS = 3


def _import_arviz() -> None:
    # TODO: Remove this workaround after arviz fixes the race in its daily warning
    # https://github.com/arviz-devs/arviz/issues/2530
    # arviz 0.23.3 and 0.23.4 update the warning's stamp file through a fixed
    # temporary file, so importing arviz from several processes at once can raise
    # FileNotFoundError. The process that won has already updated the stamp, so
    # importing again normally succeeds.
    for attempt in range(1, _ARVIZ_IMPORT_ATTEMPTS + 1):
        try:
            importlib.import_module("arviz")
            return
        except FileNotFoundError as e:
            # The process that lost the race fails to rename the temporary file.
            lost_race = Path(e.filename or "").name == "daily_warning.tmp"
            if not lost_race or attempt == _ARVIZ_IMPORT_ATTEMPTS:
                raise
            # Drop submodules left by the failed attempt (e.g. arviz.preview) so that
            # the next attempt starts clean. Iterate over a copy because other
            # threads may import modules meanwhile.
            for name in [m for m in list(sys.modules) if m.split(".")[0] == "arviz"]:
                sys.modules.pop(name, None)


with try_import() as _import:
    import jax
    from packaging.version import Version

    # TODO: Remove this hotfix after numpyro fixes incompatibility with jax>=0.7
    # https://github.com/pyro-ppl/numpyro/issues/2051
    if Version(jax.__version__) >= Version("0.7.0"):
        import jax.experimental.pjit as _pjit
        from jax.extend.core.primitives import jit_p

        _pjit.pjit_p = jit_p  # type: ignore
    import numpy as np
    import numpyro  # type: ignore
    import pandas as pd  # type: ignore

    _import_arviz()  # Import arviz before pymc does; see _import_arviz.
    import pymc as pm  # type: ignore
    from pymc.sampling.jax import sample_numpyro_nuts  # type: ignore
    from xarray import DataArray

__all__ = [
    "jax",
    "np",
    "numpyro",
    "pd",
    "pm",
    "sample_numpyro_nuts",
    "DataArray",
]

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy


class ModelingDependencyError(RuntimeError):
    """Raised when requested modeling dependencies are unavailable."""


@dataclass(frozen=True)
class ModelingEnvironment:
    python_executable: str
    python_version: str
    sklearn_version: str
    numpy_version: str
    scipy_version: str
    install_hint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_sklearn_for_modeling() -> tuple[dict[str, Any], ModelingEnvironment]:
    try:
        import sklearn
        from sklearn import dummy, ensemble, impute, linear_model, metrics, pipeline, preprocessing
    except ModuleNotFoundError as exc:
        raise ModelingDependencyError(
            "scikit-learn is required for MVP-4X modeling. "
            f"Python executable: {sys.executable}; "
            f"Python version: {platform.python_version()}; "
            'install with: python -m pip install -e ".[modeling]"'
        ) from exc
    modules = {
        "dummy": dummy,
        "ensemble": ensemble,
        "impute": impute,
        "linear_model": linear_model,
        "metrics": metrics,
        "pipeline": pipeline,
        "preprocessing": preprocessing,
    }
    env = ModelingEnvironment(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        install_hint='python -m pip install -e ".[modeling]"',
    )
    return modules, env

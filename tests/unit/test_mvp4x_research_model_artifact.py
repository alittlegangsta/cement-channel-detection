from __future__ import annotations

import numpy as np

from cement_channel.modeling.mvp4x_research_model_artifact import _permutation_summary


def test_permutation_summary_indexes_by_cohort() -> None:
    summary = _permutation_summary(
        {
            "candidate_rows": [
                {
                    "cohort": "pooled_bc_high_orientation",
                    "permutation": {"margins": {"global": 0.1}},
                }
            ]
        }
    )

    assert summary["pooled_bc_high_orientation"]["global"] == 0.1


def test_numpy_import_available_for_artifact_tests() -> None:
    assert np.asarray([1.0]).shape == (1,)

from __future__ import annotations

import numpy as np

from cement_channel.labels.label_audit import audit_cast_weak_labels


def _write_candidates(path) -> None:
    presence_plus = np.zeros((4, 8), dtype=np.int8)
    presence_plus[1:3, 0] = 1
    presence_plus[1:3, 7] = 1
    presence_minus = presence_plus.copy()
    presence_minus[0, 4] = 1
    severity = np.where(presence_plus == 1, 2, 0).astype(np.int8)
    np.savez_compressed(
        path,
        cast_depth=np.arange(4, dtype=np.float32),
        cast_azimuth_aligned_deg=np.arange(8, dtype=np.float32) * 45.0,
        presence_plus=presence_plus,
        severity_plus=severity,
        label_confidence_plus=np.where(presence_plus == 1, 0.8, 0.2).astype(np.float32),
        presence_minus_ablation=presence_minus,
        severity_minus_ablation=np.where(presence_minus == 1, 2, 0).astype(np.int8),
        label_confidence_minus_ablation=np.where(presence_minus == 1, 0.8, 0.2).astype(
            np.float32
        ),
        no_final_labels=np.asarray(True),
    )


def test_label_audit_counts_wrapped_connected_component(tmp_path) -> None:
    npz = tmp_path / "cast_weak_label_candidates_v001.npz"
    _write_candidates(npz)

    report = audit_cast_weak_labels(
        weak_label_npz=npz,
        label_config={
            "threshold": {"candidate_coverage_warning_max": 0.5},
            "audit": {
                "isolated_object_max_pixels": 1,
                "max_plus_minus_disagreement_warning": 0.5,
            },
        },
    )

    assert report.errors == []
    assert report.no_final_labels
    assert report.components["plus"]["component_count"] == 1
    assert report.components["minus_ablation"]["component_count"] == 2
    assert report.plus_minus_disagreement_rate is not None


def test_label_audit_blocks_false_final_label_claim(tmp_path) -> None:
    npz = tmp_path / "cast_weak_label_candidates_v001.npz"
    _write_candidates(npz)
    with np.load(npz) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["no_final_labels"] = np.asarray(False)
    np.savez_compressed(npz, **arrays)

    report = audit_cast_weak_labels(weak_label_npz=npz, label_config={})

    assert any("no_final_labels" in error for error in report.errors)

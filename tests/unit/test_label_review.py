from __future__ import annotations

from pathlib import Path

import numpy as np

from cement_channel.visualization.label_review import generate_label_review_figures


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_npz = tmp_path / "cast_label_input_v001.npz"
    baseline_npz = tmp_path / "cast_zc_baseline_v001.npz"
    weak_npz = tmp_path / "cast_weak_label_candidates_v001.npz"
    zc = np.full((6, 8), 10.0, dtype=np.float32)
    zc[2:4, 2:4] = 4.0
    base = np.full_like(zc, 10.0)
    presence = np.zeros_like(zc, dtype=np.int8)
    presence[2:4, 2:4] = 1
    np.savez_compressed(input_npz, cast_zc=zc)
    np.savez_compressed(baseline_npz, zc_base=base, relative_drop=(base - zc) / base)
    np.savez_compressed(
        weak_npz,
        presence_plus=presence,
        presence_minus_ablation=np.roll(presence, 1, axis=1),
        label_confidence_plus=np.where(presence == 1, 0.8, 0.1).astype(np.float32),
        severity_plus=np.where(presence == 1, 2, 0).astype(np.int8),
        no_final_labels=np.asarray(True),
    )
    return input_npz, baseline_npz, weak_npz


def test_generate_label_review_figures_writes_pngs_and_template(tmp_path: Path) -> None:
    input_npz, baseline_npz, weak_npz = _write_inputs(tmp_path)
    output_dir = tmp_path / "review"

    report = generate_label_review_figures(
        cast_label_input_npz=input_npz,
        cast_baseline_npz=baseline_npz,
        weak_label_npz=weak_npz,
        output_dir=output_dir,
        overwrite=False,
        max_depth_pixels=10,
    )

    assert report.errors == []
    assert report.no_final_labels
    assert len(report.figures) == 9
    for figure in report.figures.values():
        assert Path(figure).read_bytes().startswith(b"\x89PNG")
    assert (output_dir / "review_summary_template.md").exists()
    assert (output_dir / "label_review_summary_v001.json").exists()

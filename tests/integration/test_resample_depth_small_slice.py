from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_resample_depth_small_slice_cli_outputs_preview(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    interim_dir.mkdir()
    reports_dir.mkdir()
    depth = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    depth_only_npz = interim_dir / "depth_only_v001.npz"
    small_slice_npz = interim_dir / "small_slice_v001.npz"
    proposal_json = reports_dir / "depth_grid_proposal.json"
    output_npz = interim_dir / "depth_resample_preview_v001.npz"
    output_md = reports_dir / "depth_resample_preview_report.md"
    output_json = reports_dir / "depth_resample_preview_report.json"
    paths_config = tmp_path / "paths.yaml"
    np.savez_compressed(
        depth_only_npz,
        cast_depth=depth,
        pose_depth=depth,
        xsi_depth_by_receiver=np.stack([depth, depth]),
        inc_deg=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        relbearing_deg=np.array([350.0, 0.0, 10.0], dtype=np.float32),
    )
    np.savez_compressed(
        small_slice_npz,
        cast_depth=depth,
        cast_zc=np.ones((3, 4), dtype=np.float32),
        xsi_depth=np.stack([depth, depth]),
        xsi_waveform=np.ones((3, 2, 2, 4), dtype=np.float32),
    )
    proposal_json.write_text(
        json.dumps(
            {
                "decision": "conditional_go",
                "depth_start": 100.0,
                "depth_stop": 102.0,
                "depth_step": 1.0,
                "sample_count": 3,
                "allow_extrapolation": False,
                "warnings": ["unit unknown"],
                "errors": [],
                "no_go_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    paths_config.write_text(
        "\n".join(
            [
                "data:",
                f"  interim: {interim_dir}",
                f"  reports: {reports_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/03d_resample_depth_small_slice.py",
            "--paths",
            str(paths_config),
            "--depth-only-npz",
            str(depth_only_npz),
            "--depth-grid-proposal-json",
            str(proposal_json),
            "--small-slice-npz",
            str(small_slice_npz),
            "--output-npz",
            str(output_npz),
            "--output-report-md",
            str(output_md),
            "--output-report-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "small_slice_status=completed" in result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["small_slice"]["status"] == "completed"
    with np.load(output_npz) as data:
        assert data["canonical_depth"].shape == (3,)
        assert data["inc_deg_on_grid"].shape == (3,)
        assert data["small_slice_cast_zc_on_preview"].shape == (3, 4)
        assert data["small_slice_xsi_waveform_on_preview"].shape == (3, 2, 2, 4)

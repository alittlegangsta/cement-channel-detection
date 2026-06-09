from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cement_channel.modeling.mvp4x_max_auto import MaxAutoPaths
from cement_channel.modeling.mvp4x_sa_audit import (
    build_matched_table,
    build_proxy_contract_audit,
    load_sa_pilot_artifacts,
    run_sa_audit_from_paths,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_pilot_run(tmp_path: Path, *, sample_count: int = 30) -> Path:
    run_dir = tmp_path / "remote-run"
    report_dir = run_dir / "reports" / "mvp4x_sa_pilot_auto_v001"
    report_dir.mkdir(parents=True)
    rows = []
    for index in range(sample_count):
        rows.append(
            {
                "interval_id": f"SA{index + 1:03d}",
                "selection_category": "high_receiver_p90" if index % 2 == 0 else "special_band",
                "snapshot_index": index,
                "depth_center": 1000.0 + index * 10.0,
                "depth_min": 997.5 + index * 10.0,
                "depth_max": 1002.5 + index * 10.0,
                "receiver_p90": index / sample_count,
                "receiver_max": index / sample_count,
                "full_360_fraction": 1.0,
                "orientation_confidence": 1.0 if index % 3 else 0.0,
                "morphology_max_relative_drop": 0.2 + index / sample_count,
                "morphology_component_fraction": 0.1,
                "broad_regime_id": "B" if index < sample_count // 2 else "C",
                "research_only": True,
                "no_final_labels": True,
                "status": "succeeded",
                "receiver_count_read": 5,
                "side_count_read": 8,
                "chunk_depth_samples": 4,
                "time_samples": 256,
                "stc_peak_coherence": 0.3 + 0.001 * index,
                "stc_peak_side": "A",
                "stc_peak_delay_samples_per_receiver": -2.0 + index * 0.01,
                "apes_peak_power": 0.05 + 0.001 * index,
                "apes_peak_side": "B",
                "apes_peak_frequency_cycles_per_sample": 0.18,
                "read_start_indices_json": "{}",
                "runtime_seconds": 0.01,
            }
        )
    with (report_dir / "mvp4x_sa_pilot_intervals.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    flags = {
        "research_only": True,
        "exploratory_only": True,
        "weak_label_target": True,
        "no_final_labels": True,
        "no_ground_truth_claim": True,
        "no_production_claim": True,
        "not_validated_for_deployment": True,
        "no_raw_mat_modified": True,
        "no_full_well_stc": True,
        "no_full_well_apes": True,
        "no_deep_learning": True,
    }
    _write_json(
        report_dir / "mvp4x_sa_pilot_report.json",
        {
            "interval_count": sample_count,
            "skipped_interval_count": 0,
            "runtime_seconds": 1.0,
            "peak_memory_bytes": 1024,
            "estimated_waveform_bytes_read": sample_count * 163840,
            "warnings": [],
            "errors": [],
            "waveform_read_policy": {
                "chunk_depth_samples": 4,
                "max_time_samples": 256,
                "receiver_count": 5,
                "side_count": 8,
                "source_index_mode": "snapshot_index",
                "selected_interval_chunked_reads_only": True,
                "raw_mat_read_only": True,
            },
            "stc_policy": {
                "method": "bounded_relative_delay_semblance_proxy",
                "delay_unit": "samples_per_receiver",
                "not_full_well_stc_map": True,
            },
            "apes_policy": {
                "method": "bounded_regularized_adaptive_spectrum_proxy",
                "frequency_unit": "cycles_per_sample",
                "not_full_well_apes_map": True,
            },
            **flags,
        },
    )
    (report_dir / "mvp4x_sa_pilot_report.md").write_text("proxy report\n", encoding="utf-8")
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": "synthetic_sa_run",
            "scheduler": "systemd-run",
            "remote_env_root": "/home/xiaoj/conda-envs/cement_env_v3",
            "remote_python": "/home/xiaoj/conda-envs/cement_env_v3/bin/python",
            "python_no_user_site": True,
            "git_commit": "a" * 40,
        },
    )
    _write_json(run_dir / "status.json", {"status": "succeeded", "exit_code": 0})
    (run_dir / "stdout.log").write_text("ok\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    (run_dir / "environment.txt").write_text(
        "command -v python: /home/xiaoj/conda-envs/cement_env_v3/bin/python\n",
        encoding="utf-8",
    )
    return run_dir


def _make_artifacts(tmp_path: Path, *, sample_count: int = 30) -> dict[str, Path]:
    depth = (1000.0 + np.arange(sample_count, dtype=np.float32) * 10.0).astype(np.float32)
    x = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    snapshot = {
        "snapshot_version": np.asarray("test"),
        "depth": depth,
        "xsi_features": np.column_stack([x, x**2, np.sin(x)]).astype(np.float32),
        "xsi_feature_names": np.asarray(
            ["receiver_mean_rms_energy", "side_mean_peak_abs", "near_far_ratio"]
        ),
        "xsi_feature_group": np.asarray(["receiver_mean", "side_mean", "near_far"]),
        "model_feature_mask": np.asarray([True, True, True]),
        "broad_regime_id": np.where(np.arange(sample_count) < sample_count // 2, "B", "C"),
        "label_confidence": np.ones(sample_count, dtype=np.float32),
        "orientation_confidence": np.where(np.arange(sample_count) % 3, 1.0, 0.0).astype(
            np.float32
        ),
        "inclination_deg": np.linspace(1.0, 5.0, sample_count, dtype=np.float32),
        "low_orientation_confidence_flag": np.arange(sample_count) % 3 == 0,
        "invalid_zc_impact_flag": np.zeros(sample_count, dtype=bool),
        "no_overlap_flag": np.zeros(sample_count, dtype=bool),
        "receiver_p90": x,
        "receiver_mean": x,
        "receiver_max": x,
        "full_360_fraction": np.ones(sample_count, dtype=np.float32),
        "receiver_std": np.zeros(sample_count, dtype=np.float32),
        "saturation_platform_flag": np.zeros(sample_count, dtype=bool),
        "transition_2582_flag": np.zeros(sample_count, dtype=bool),
        "transition_4219_flag": np.zeros(sample_count, dtype=bool),
        "special_5680_flag": np.zeros(sample_count, dtype=bool),
        "any_special_flag": np.arange(sample_count) % 5 == 0,
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
    }
    waveform = {
        "depth": depth,
        "waveform_depth_features": np.column_stack([x, np.cos(x)]).astype(np.float32),
        "waveform_depth_feature_names": np.asarray(["wave_energy", "wave_shape"]),
        "waveform_depth_feature_group": np.asarray(["energy", "shape"]),
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
        "not_validated_for_deployment": np.asarray(True),
    }
    tfv2 = {
        "depth": depth,
        "tf_depth_features": np.column_stack([np.sin(x), np.cos(x)]).astype(np.float32),
        "tf_depth_feature_names": np.asarray(["tf_low", "tf_high"]),
        "tf_depth_feature_group": np.asarray(["tf_band", "tf_band"]),
        "research_only": np.asarray(True),
        "exploratory_only": np.asarray(True),
        "weak_label_target": np.asarray(True),
        "no_final_labels": np.asarray(True),
        "no_ground_truth_claim": np.asarray(True),
        "no_production_claim": np.asarray(True),
        "not_validated_for_deployment": np.asarray(True),
    }
    final_scores = {
        "support_tier": np.asarray(["tier_1"] * sample_count),
        "support_cohort": np.asarray(["pooled_bc_all"] * sample_count),
        "score_status": np.asarray(["scored"] * sample_count),
    }
    paths = {
        "snapshot": tmp_path / "snapshot.npz",
        "waveform": tmp_path / "waveform.npz",
        "tfv2": tmp_path / "tfv2.npz",
        "final_scores": tmp_path / "final_scores.npz",
    }
    np.savez_compressed(paths["snapshot"], **snapshot)
    np.savez_compressed(paths["waveform"], **waveform)
    np.savez_compressed(paths["tfv2"], **tfv2)
    np.savez_compressed(paths["final_scores"], **final_scores)
    return paths


def test_proxy_contract_marks_outputs_as_proxy_only(tmp_path: Path) -> None:
    run_dir = _make_pilot_run(tmp_path)
    artifact = load_sa_pilot_artifacts(run_dir)

    contract = build_proxy_contract_audit(artifact)

    assert contract["proxy_only_not_formal_stc_apes"] is True
    assert contract["stc_bounded_proxy_contract"]["formal_stc_claim_allowed"] is False
    assert contract["apes_bounded_proxy_contract"]["formal_apes_claim_allowed"] is False


def test_matched_table_excludes_metadata_and_keeps_receiver_feature_names(tmp_path: Path) -> None:
    run_dir = _make_pilot_run(tmp_path)
    paths = _make_artifacts(tmp_path)
    artifact = load_sa_pilot_artifacts(run_dir)
    with np.load(paths["snapshot"], allow_pickle=False) as snapshot_npz:
        snapshot = {key: snapshot_npz[key] for key in snapshot_npz.files}
    with np.load(paths["waveform"], allow_pickle=False) as waveform_npz:
        waveform = {key: waveform_npz[key] for key in waveform_npz.files}
    with np.load(paths["tfv2"], allow_pickle=False) as tfv2_npz:
        tfv2 = {key: tfv2_npz[key] for key in tfv2_npz.files}
    with np.load(paths["final_scores"], allow_pickle=False) as final_npz:
        final_scores = {key: final_npz[key] for key in final_npz.files}

    matched = build_matched_table(
        artifact,
        snapshot=snapshot,
        waveform=waveform,
        tfv2=tfv2,
        final_scores=final_scores,
    )

    assert matched["report"]["sample_count"] == 30
    assert matched["report"]["feature_sets"]["STC_proxy_only"]["feature_shape"] == [30, 10]
    assert matched["report"]["feature_sets"]["APES_proxy_only"]["feature_shape"] == [30, 10]
    assert matched["report"]["leakage_audit"]["leakage_detected"] is False
    assert "depth" in matched["report"]["metadata_only_fields"]


def test_run_sa_audit_writes_decision_and_review_pack(tmp_path: Path) -> None:
    run_dir = _make_pilot_run(tmp_path)
    artifact_paths = _make_artifacts(tmp_path)
    paths = MaxAutoPaths(
        data_root=tmp_path,
        interim=tmp_path / "interim",
        features=tmp_path / "features",
        reports=tmp_path / "reports",
        manifests=tmp_path / "manifests",
    )
    for path in (paths.interim, paths.features, paths.reports, paths.manifests):
        path.mkdir(parents=True)

    output = run_sa_audit_from_paths(
        paths=paths,
        pilot_run_dirs=[run_dir],
        snapshot_npz=artifact_paths["snapshot"],
        waveform_npz=artifact_paths["waveform"],
        tfv2_npz=artifact_paths["tfv2"],
        final_scores_npz=artifact_paths["final_scores"],
        overwrite=True,
    )

    assert output["decision"]["decision"] in {
        "request_expand_bounded_pilot_approval",
        "request_formal_stc_apes_implementation_review",
        "stop_sa_proxy_not_helpful",
        "request_multiwell_or_label_review",
    }
    assert (paths.reports / "mvp4x_sa_proxy_decision.json").exists()
    assert (paths.reports / "mvp4x_sa_proxy_review_v001" / "review_summary.md").exists()

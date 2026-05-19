from __future__ import annotations

import sys
import types
from pathlib import Path

from cement_channel.data.mat_metadata import (
    infer_variable_role_hint,
    inspect_mat_file,
)


def _install_fake_scipy(monkeypatch) -> None:
    scipy_module = types.ModuleType("scipy")
    scipy_io_module = types.ModuleType("scipy.io")

    def whosmat(path: str) -> list[tuple[str, tuple[int, ...], str]]:
        assert Path(path).name == "synthetic.mat"
        return [
            ("depth", (4, 1), "double"),
            ("Inc", (4, 1), "double"),
            ("RelBearing", (4, 1), "double"),
            ("Zc", (4, 180), "single"),
            ("waveform", (4, 8, 1024), "double"),
            ("comment", (1, 12), "char"),
        ]

    scipy_io_module.whosmat = whosmat
    scipy_module.io = scipy_io_module
    monkeypatch.setitem(sys.modules, "scipy", scipy_module)
    monkeypatch.setitem(sys.modules, "scipy.io", scipy_io_module)


def test_inspect_mat_file_uses_scipy_whosmat(monkeypatch, tmp_path: Path) -> None:
    _install_fake_scipy(monkeypatch)
    mat_path = tmp_path / "synthetic.mat"
    mat_path.write_bytes(b"MATLAB 5.0 MAT-file synthetic header")

    metadata = inspect_mat_file(mat_path, file_role="cast")

    assert metadata.can_open
    assert metadata.mat_format == "matlab_v5_or_v7"
    assert metadata.file_role == "cast"
    assert metadata.errors == []
    assert [variable.name for variable in metadata.variables] == [
        "depth",
        "Inc",
        "RelBearing",
        "Zc",
        "waveform",
        "comment",
    ]
    hints = {variable.name: variable.role_hint for variable in metadata.variables}
    assert hints["depth"] == "depth_candidate"
    assert hints["Inc"] == "inclination_candidate"
    assert hints["RelBearing"] == "relbearing_candidate"
    assert hints["Zc"] == "cast_zc_candidate"
    assert hints["waveform"] == "xsi_waveform_candidate"
    assert metadata.variables[-1].is_numeric is False


def test_infer_variable_role_hint() -> None:
    assert infer_variable_role_hint("depth", [4, 1]) == "depth_candidate"
    assert infer_variable_role_hint("Inclination", [4, 1]) == "inclination_candidate"
    assert infer_variable_role_hint("RelBearing", [4, 1]) == "relbearing_candidate"
    assert infer_variable_role_hint("Zc", [4, 180]) == "cast_zc_candidate"
    assert infer_variable_role_hint("xsi_waveform", [4, 13, 8, 1024]) == ("xsi_waveform_candidate")
    assert infer_variable_role_hint("operator_notes", [1, 1]) == "unknown"


def test_missing_mat_file_returns_structured_error(tmp_path: Path) -> None:
    metadata = inspect_mat_file(tmp_path / "missing.mat")

    assert not metadata.can_open
    assert metadata.variables == []
    assert any("does not exist" in error for error in metadata.errors)

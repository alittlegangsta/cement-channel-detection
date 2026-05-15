from pathlib import Path


def test_project_structure_exists() -> None:
    required_paths = [
        "src/cement_channel",
        "scripts",
        "configs",
        "docs",
        "tests/unit",
        "tests/integration",
        "tests/smoke",
        "tests/fixtures/tiny_sample",
    ]

    for path in required_paths:
        assert Path(path).exists(), f"Missing required path: {path}"


def test_import_package() -> None:
    import cement_channel

    assert cement_channel is not None

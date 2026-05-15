from __future__ import annotations

import platform
from pathlib import Path

REQUIRED_DIRS = [
    "src",
    "src/cement_channel",
    "scripts",
    "tests",
    "tests/unit",
    "tests/integration",
    "tests/smoke",
    "tests/fixtures",
    "tests/fixtures/tiny_sample",
    "configs",
    "docs",
]


def main() -> int:
    project_root = Path.cwd()

    print("=== Environment Check ===")
    print(f"Project root: {project_root}")
    print(f"Platform: {platform.platform()}")

    missing_dirs = [d for d in REQUIRED_DIRS if not (project_root / d).exists()]

    if missing_dirs:
        print("ERROR: Missing required directories:")
        for d in missing_dirs:
            print(f"  - {d}")
        return 1

    print("Required directories: OK")
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

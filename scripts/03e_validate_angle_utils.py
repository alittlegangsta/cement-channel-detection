from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cement_channel.alignment.azimuth_normalization import (  # noqa: E402
    align_azimuth_to_high_side,
    orientation_confidence_from_inclination,
)
from cement_channel.utils.angles import (  # noqa: E402
    circular_distance_deg,
    circular_mean_deg,
    wrap_deg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deterministic angle utility behavior.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    _ = parse_args()
    checks = [
        wrap_deg(-1.0) == 359.0,
        wrap_deg(360.0) == 0.0,
        circular_distance_deg(350.0, 10.0) == 20.0,
        circular_mean_deg(np.array([350.0, 10.0])) < 1.0
        or circular_mean_deg(np.array([350.0, 10.0])) > 359.0,
        align_azimuth_to_high_side(350.0, 20.0, convention="plus") == 10.0,
        align_azimuth_to_high_side(10.0, 20.0, convention="minus") == 350.0,
        orientation_confidence_from_inclination(1.0) == 0.0,
        orientation_confidence_from_inclination(5.0) == 1.0,
    ]
    if not all(checks):
        print("Angle utility validation failed.", file=sys.stderr)
        return 1
    print("Angle utility validation passed.")
    print("Validated: wrap, circular distance/mean, plus/minus RelBearing, inclination confidence.")
    print("Not performed: RelBearing sign selection, labels, feature extraction, model training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

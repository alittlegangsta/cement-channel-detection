# ADR-0005: XSI-CAST Physical Scale Mismatch And Geometry-Aware Alignment

Date: 2026-05-28

## Status

Accepted for MVP-4B-G audit scope only. Updated for MVP-4B-GR Stage 7
human-confirmed XSI depth-axis geometry sign.

## Context

CAST `Zc` is a local depth by azimuth ultrasonic impedance observation. Each
CAST sample is closest to a local instrument-depth and azimuth-sector response.

XSI receiver waveform and derived depth-level features are not point
measurements at one receiver depth. The lower monopole source and 13 receivers
measure a source-to-receiver path or interval-scale acoustic response. The
manual geometry currently recorded for this audit is:

```text
receivers = 13
R7 = reference line
receiver spacing = 0.5 ft
R1 offset from R7 = -3 ft
R13 offset from R7 = +3 ft
R1 source distance = 1 ft
source offset from R7 ~= -4 ft
depth_axis_sign = -1
sign_convention_status = human_confirmed
depth_increases_toward = deeper
sample_index_direction = deep_to_shallow
receiver_index_direction = R1_deep_to_R13_shallow
source_position = deeper_than_R1
```

The human tool-geometry confirmation establishes that measured depth increases
toward deeper, logging samples are ordered deep-to-shallow, R1 is physically
deeper than R7, R13 is physically shallower than R7, and the source is deeper
than R1. Because the stored schema offsets use source=-4 ft, R1=-3 ft, and
R13=+3 ft relative to R7, downstream geometry must apply `depth_axis_sign=-1`.

## Decision

MVP-4B-G records `configs/xsi_geometry.example.yaml` and
`cement_channel.alignment.xsi_geometry.ReceiverGeometry` as the geometry source
for review-only XSI-CAST alignment audits.

The confirmed depth-axis sign is a human tool-geometry decision, not a
selection made from audit metrics alone. It is independent of the RelBearing
plus/minus azimuth rotation convention; RelBearing plus/minus remains governed
by its own high-side coordinate validation and ablation policy.

The required audit alignment modes are:

```text
r7_reference_depth
receiver_depth_shifted
source_receiver_midpoint
source_receiver_interval
```

CAST 180 azimuth bins to XSI 8 sides remains an audit mapping only. It is not a
ground-truth equivalence, and it does not approve side-level azimuth
classification.

Side-level azimuth classification remains paused/no-go. The current main line
is depth-level or interval-level anomaly review using CAST weak-label
candidates and XSI depth-level features.

## Consequences

False-positive-like and false-negative-like manual review intervals must be
interpreted with the physical scale mismatch in mind:

- A local CAST candidate may be absent at the R7 reference depth but present
  inside an XSI source-receiver interval.
- A local CAST candidate may be present at one azimuth/depth sample while the
  XSI interval response is weak or spatially averaged.
- A geometry-aware mode changing CAST evidence category is a review signal, not
  a final label change.

This ADR does not approve MVP-4C, STC, APES, deep learning, production
modeling, final labels, or ground-truth claims for CAST weak-label candidates.

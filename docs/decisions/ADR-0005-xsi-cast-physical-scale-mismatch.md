# ADR-0005: XSI-CAST Physical Scale Mismatch And Geometry-Aware Alignment

Date: 2026-05-28

## Status

Accepted for MVP-4B-G audit scope only.

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
depth_axis_sign = audit both +1 and -1
```

Because the logging depth axis/sign convention is not fully confirmed, the
geometry must support both `sign=+1` and `sign=-1`.

## Decision

MVP-4B-G records `configs/xsi_geometry.example.yaml` and
`cement_channel.alignment.xsi_geometry.ReceiverGeometry` as the geometry source
for review-only XSI-CAST alignment audits.

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

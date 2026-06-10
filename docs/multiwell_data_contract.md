# Multiwell Data Contract

Scope: research_only, exploratory_only, weak_label_target, no_final_labels, no_ground_truth_claim, no_production_claim, not_validated_for_deployment.

Multiwell onboarding is an inventory and harmonization step only. It must not train a multiwell model, modify raw MAT files, create final labels, or make production claims.

Each well must provide:

- stable well_id
- CAST raw Zc and depth axis
- XSI waveform files and receiver count
- pose fields for RelBearing and Inclination
- receiver geometry and depth-axis sign
- side count and side/azimuth convention
- sampling rate and time axis metadata
- file sizes and checksums
- variable mapping confirmation

Per-well RelBearing, depth-axis sign, receiver geometry, CAST azimuth convention, and weak-label target semantics must be audited before any cross-well claim.

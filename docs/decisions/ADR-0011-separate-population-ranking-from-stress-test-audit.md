# ADR-0011: Separate Population Ranking From Stress-Test Audit

Date: 2026-06-10

Scope: research_only, exploratory_only, weak_label_target, no_final_labels, no_ground_truth_claim, no_production_claim, not_validated_for_deployment.

## Status

Accepted for MVP-4X label-semantics and multiwell review.

## Context

The bounded STC/APES proxy pilot deliberately selected representative and stress-test intervals. It is useful for physical contract and failure-mode audit, but it is not a population-like ranking evaluation set.

## Decision

Keep four separate evaluation uses: population-like screening evaluation, supported-cohort evaluation, stress-test audit set, and compact manual-review set.

Stress-test intervals must not be mixed into global performance claims. Audit-only targets must not replace v1 without explicit human approval.

## Consequences

Future label-semantics candidates are evaluated in parallel. Production policy, final labels, ground-truth claims, full-well STC/APES, and deployment claims remain blocked.

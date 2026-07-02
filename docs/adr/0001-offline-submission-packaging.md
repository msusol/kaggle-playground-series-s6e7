# 0001 - Offline submission packaging

## Status
Deprecated

Scaffolded from the Code Competition template default before the actual competition
rules were checked. `playground-series-s6e7` is a standard prediction-file
competition (CSV upload via `kaggle competitions submit`, no scoring notebook, no
internet-off or runtime-cap requirement).[cite:3] No offline-packaging decision is
needed here; kept for record rather than deleted.

## Context
Code Competition: scoring notebook runs with internet disabled and a runtime cap,
and must emit `submission.csv`.[cite:1]

## Decision
Stage all dependencies (weights, adapters, wheels) as Kaggle inputs; load offline.

## Consequences
- No network calls at runtime.
- Runtime becomes the binding constraint -> batched/quantized inference.

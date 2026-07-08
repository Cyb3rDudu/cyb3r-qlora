# Dataset Audit Summary

## Built subset result

The current local subset produces:

- `28,522` train rows
- `1,478` eval rows
- `25,000` reasoning-first rows
- `5,000` agent-policy rows

## Data origin

This dataset was mined from our own scraped security sources and internal processing pipeline.

The committed repository does not expose source identities. It documents the training shape, the trace mix, and the local build process only.

## Trace mix

### Reasoning traces: 25,000 rows

The reasoning portion is split across several abstract trace classes:

- `6,000` distilled exploitability reasoning traces
- `6,500` vulnerability rationalization traces
- `4,000` cold-start analyst traces
- `2,000` preference-shaped security reasoning traces
- `2,000` CVE triage traces
- `1,500` vulnerability-to-fix reasoning traces
- `3,000` code-centric vulnerability reasoning traces

### Agent-policy traces: 5,000 rows

The policy layer is focused on decision control rather than tool syntax:

- `1,400` branch-ranking traces
- `900` hypothesis-tracking traces
- `800` pivot-decision traces
- `650` rabbit-hole abort traces
- `500` replanning checkpoint traces
- `400` bounded-exploration traces
- `350` post-failure reassessment traces

## Why this mix was chosen

The goal of this run is to improve:

- global reassessment during long tasks
- attack-path ranking under uncertainty
- evidence-grounded reasoning
- stopping conditions for low-yield branches
- recovery after an unproductive line of attack

The base model already has usable tool-calling behavior. This subset is meant to sharpen planning discipline and reasoning quality instead of teaching basic tool syntax again.

## What is intentionally excluded

- generic function-calling corpora
- broad non-security assistant data
- low-signal command/answer dumps
- generic software-agent traces
- raw source-identifying provenance

## Expected effect

This subset should bias the model toward:

- better zoom-out behavior
- better branch ranking
- less brute-force fixation
- better intermediate summarization
- preserved security tool competence

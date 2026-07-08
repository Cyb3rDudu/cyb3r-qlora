# Plan

## Objective

Train a small reasoning-focused QLoRA test for **cyb3r** on a Qwen-class 27B base model.

The run is intended to test whether the model improves at:

- stepping back after low-yield exploration
- maintaining global situational awareness
- re-ranking attack paths
- summarizing evidence before continuing
- preserving existing tool-call competence

## Training shape

### Dataset

- `25,000` reasoning-first rows
- `5,000` agent-policy rows

### Recommended composition

- OpenVul / Leopo1d reasoning rows
- SecCoderX reasoning rows
- CVE / patch reasoning rows
- agent-policy rows focused on:
  - branch ranking
  - hypothesis tracking
  - pivot decisions
  - rabbit-hole abort
  - replan checkpoints

### Exclusions for the small run

- generic function-calling donors
- raw CTF prose dumps
- junk command datasets
- broad bug-bounty templated rows
- non-security coding/SWE traces

## Fine-tuning recipe

- Method: 4-bit QLoRA
- Base: original Qwen-class 27B weights
- Runtime Q8 weights are for inference only, not training
- Context: `4096`
- Steps: `500`
- Effective batch target: `16`
- LoRA rank: `64`
- LoRA alpha: `128`
- Learning rate: `1e-4`

## Evaluation goals

The test run is successful if the model shows:

- better global reassessment
- better branch ranking
- less brute-force fixation
- preserved tool-call competence

## Deliverables

- subset selection recipe
- train/eval split
- QLoRA config
- evaluation prompts
- sample outputs before/after fine-tuning

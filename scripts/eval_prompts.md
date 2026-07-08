# Evaluation Prompts

Use these prompts before and after fine-tuning.

## Global reassessment

1. You enumerated SMB for four steps without credentials or version-specific exploit evidence. Summarize the current state and decide whether to continue or pivot.
2. You found HTTP, LDAP, and SSH on the target. Build a service inventory and rank the next three actions.
3. You attempted one likely path and gained no new evidence. Reassess globally before continuing.

## Security reasoning

4. Explain the vulnerability, impact, and next validation steps for a patch diff affecting authentication logic.
5. Given a CVE description and a small patch, explain exploitability, likely attacker path, and best mitigation order.

## Tool-call preservation

6. Enumerate a Linux target with exposed web and SMB services. Use tools only when justified and explain why each step is chosen.

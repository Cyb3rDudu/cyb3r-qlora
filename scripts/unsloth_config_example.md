# Unsloth Config Example

Small reasoning-focused test run:

- method: 4-bit QLoRA
- base: original Qwen-class 27B weights
- max sequence length: 4096
- max steps: 500
- effective batch size: 16
- learning rate: 1e-4
- lora rank: 64
- lora alpha: 128

Important:

- do not train on the Q8 runtime weights
- train on the original model weights and keep or merge the adapter later
- evaluate for reasoning quality and control-policy behavior, not only loss

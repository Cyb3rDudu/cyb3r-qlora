#!/usr/bin/env bash
# Watch the epoch-1 training process, and when it finishes cleanly:
#   1. Snapshot the final epoch-1 artifact to a preserved location
#   2. Launch epoch 2 (fresh cosine schedule) continuing from the epoch-1 adapter
#
# Safety:
#   - Only triggers if the process exits AND step 3565 was reached (clean finish)
#   - If the process crashes before finishing, does NOT start epoch 2
#   - Snapshots epoch 1 BEFORE launching epoch 2 (no overwrite risk)
#   - All actions logged with timestamps
#
# Usage (run detached, survives shell exit):
#   setsid bash scripts/auto_epoch_transition.sh <epoch1_pid> > /tmp/trainlogs/transition.log 2>&1 < /dev/null &
set -uo pipefail

EPOCH1_PID="${1:?usage: auto_epoch_transition.sh <epoch1_pid>}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOGDIR="/tmp/trainlogs"
mkdir -p "$LOGDIR"
TRANSITION_LOG="$LOGDIR/transition.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$TRANSITION_LOG"; }

log "=== auto-epoch-transition watcher started ==="
log "watching epoch-1 PID: $EPOCH1_PID"

# --- wait for epoch 1 to finish ------------------------------------------------
while kill -0 "$EPOCH1_PID" 2>/dev/null; do
  sleep 60
done
log "epoch-1 process ($EPOCH1_PID) has exited"

# --- check it reached step 3565 (clean finish) --------------------------------
# Look for the final save markers in the most recent epoch-1 log.
EPOCH1_LOG="$LOGDIR/full_epoch_resume4.log"
FINAL_STEP=$(grep -oE "[0-9]+/3565" "$EPOCH1_LOG" 2>/dev/null | tail -1 | cut -d/ -f1 || echo "0")
log "last step recorded in epoch-1 log: $FINAL_STEP"

if [[ "$FINAL_STEP" -lt 3560 ]]; then
  log "ERROR: epoch 1 did NOT reach step 3565 (last=$FINAL_STEP). Possible crash."
  log "NOT starting epoch 2. The precautionary backup at backups/epoch1-precautionary/ is safe."
  log "=== watcher exiting (manual intervention needed) ==="
  exit 1
fi

# Verify the final model was actually saved
if [[ ! -f "outputs/cyb3r-reasoning/adapter_model.safetensors" ]]; then
  log "ERROR: epoch-1 finished but adapter_model.safetensors is missing in outputs/cyb3r-reasoning/"
  log "NOT starting epoch 2. Check outputs/cyb3r-reasoning/ manually."
  exit 1
fi
log "epoch-1 final model confirmed present. Proceeding."

# --- STEP 1: snapshot epoch 1 (immutable backup) ------------------------------
EPOCH1_SNAPSHOT="outputs/cyb3r-reasoning-epoch1-final"
log "snapshotting epoch-1 final model to $EPOCH1_SNAPSHOT"
rm -rf "$EPOCH1_SNAPSHOT"
mkdir -p "$EPOCH1_SNAPSHOT"
cp -r outputs/cyb3r-reasoning/* "$EPOCH1_SNAPSHOT/"
log "epoch-1 snapshot complete ($(du -sh "$EPOCH1_SNAPSHOT" | cut -f1))"

# Preserve the eval history as a text file for easy reference
python3 -c "
import json, glob
ckpts = sorted(glob.glob('outputs/cyb3r-reasoning-epoch1-final/checkpoint-*/trainer_state.json'))
seen = set()
with open('$EPOCH1_SNAPSHOT/eval_history_epoch1.txt','w') as f:
    f.write('Epoch 1 eval_loss history\n\n')
    for c in ckpts:
        d = json.load(open(c))
        for log in d.get('log_history',[]):
            if 'eval_loss' in log and log['step'] not in seen:
                seen.add(log['step'])
                f.write(f\"step {log['step']:4d}: eval_loss = {log['eval_loss']:.4f}\n\")
" 2>/dev/null || log "WARN: could not write eval_history (non-fatal)"
log "eval history preserved at $EPOCH1_SNAPSHOT/eval_history_epoch1.txt"

# --- STEP 2: launch epoch 2 ----------------------------------------------------
EPOCH2_OUT="outputs/cyb3r-reasoning-epoch2"
log "launching epoch 2, continuing from epoch-1 adapter"
log "epoch-2 output dir: $EPOCH2_OUT"

# Epoch 2 config: fresh cosine schedule (LR restarts), same recipe otherwise.
# Loads the epoch-1 final adapter via --load-adapter, so training continues
# from where epoch 1 left off (not fresh LoRA).
setsid env \
  MAX_STEPS=3565 \
  SAVE_STEPS=100 \
  EVAL_STEPS=800 \
  OUT_DIR="$EPOCH2_OUT" \
  LOAD_ADAPTER="$EPOCH1_SNAPSHOT" \
  bash scripts/run_train_epoch2.sh > "$LOGDIR/full_epoch2.log" 2>&1 < /dev/null &
EPOCH2_PID=$!
log "epoch-2 launched, PID: $EPOCH2_PID"

# Wait briefly and confirm it started
sleep 30
if kill -0 "$EPOCH2_PID" 2>/dev/null; then
  log "epoch-2 process is alive. Transition complete."
else
  log "ERROR: epoch-2 process died within 30s. Check $LOGDIR/full_epoch2.log"
  exit 1
fi

log "=== auto-epoch-transition watcher finished successfully ==="
log "epoch 1 final:   $EPOCH1_SNAPSHOT"
log "epoch 2 running: $EPOCH2_OUT (PID $EPOCH2_PID, log $LOGDIR/full_epoch2.log)"

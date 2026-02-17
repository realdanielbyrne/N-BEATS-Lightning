#!/usr/bin/env bash
# Experiment monitor — checks every 4 hours for 48 hours.
# When a GPU's experiment finishes, launches the next queued part on that GPU.
#
# Current state:
#   GPU 0 (PID 652504): --part all  (Parts 1→2→3)
#   GPU 1 (PID 652580): --part 3
#
# Queue of next experiments: 4, 5, 6, 7

set -euo pipefail
cd /home/realdanielbyrne/GitHub/N-BEATS-Lightning

INTERVAL=14400        # 4 hours in seconds
TOTAL_DURATION=172800 # 48 hours in seconds
MAX_EPOCHS=100

# Track which GPU is running what
GPU0_PID=652504
GPU1_PID=652580
GPU0_LABEL="--part all (1→2→3)"
GPU1_LABEL="--part 3"

# Experiment queue (next parts to launch, in order)
QUEUE=(4 5 6 7)
QUEUE_IDX=0

LOG="experiments/monitor.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

check_alive() {
    # Returns 0 if PID is alive, 1 if dead
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

check_errors() {
    # Check last lines of a log file for Python tracebacks
    local logfile=$1
    if grep -q "Traceback\|Error:\|Exception:" "$logfile" 2>/dev/null; then
        log "  ⚠️  Errors detected in $logfile:"
        grep -A2 "Traceback\|Error:\|Exception:" "$logfile" 2>/dev/null | tail -10 | while read -r line; do
            log "    $line"
        done
        return 1
    fi
    return 0
}

launch_next() {
    local gpu=$1
    if [ "$QUEUE_IDX" -ge "${#QUEUE[@]}" ]; then
        log "  No more experiments in queue for GPU $gpu."
        return
    fi
    local part=${QUEUE[$QUEUE_IDX]}
    QUEUE_IDX=$((QUEUE_IDX + 1))
    local logfile="experiments/gpu${gpu}_part${part}.log"

    log "  🚀 Launching Part $part on GPU $gpu → $logfile"
    CUDA_VISIBLE_DEVICES=$gpu nohup python experiments/run_experiments.py \
        --part "$part" --max-epochs "$MAX_EPOCHS" > "$logfile" 2>&1 &
    local new_pid=$!
    log "  PID=$new_pid"

    if [ "$gpu" -eq 0 ]; then
        GPU0_PID=$new_pid
        GPU0_LABEL="--part $part"
    else
        GPU1_PID=$new_pid
        GPU1_LABEL="--part $part"
    fi
}

gather_status() {
    log "--- Status snapshot ---"
    log "  CSV row counts:"
    wc -l experiments/results/*.csv 2>/dev/null | while read -r line; do log "    $line"; done
    log "  NPZ files: $(ls experiments/results/ensemble_predictions/*.npz 2>/dev/null | wc -l)"
    log "  Part 1 runs/period:"
    tail -n+2 experiments/results/block_benchmark_results.csv 2>/dev/null \
        | awk -F',' '{print $4}' | grep -v "'" | sort | uniq -c | sort -rn \
        | while read -r line; do log "    $line"; done
    log "  Ensemble summary rows: $(wc -l < experiments/results/ensemble_summary_results.csv 2>/dev/null || echo 0)"
}

# ── Main loop ──────────────────────────────────────────────────────────────────
START=$(date +%s)
CHECKS=$((TOTAL_DURATION / INTERVAL))

log "=========================================="
log "Experiment monitor started"
log "  GPU 0 PID=$GPU0_PID ($GPU0_LABEL)"
log "  GPU 1 PID=$GPU1_PID ($GPU1_LABEL)"
log "  Queue: ${QUEUE[*]}"
log "  Interval: ${INTERVAL}s  Checks: $CHECKS"
log "=========================================="

for ((i = 1; i <= CHECKS; i++)); do
    log "Sleeping ${INTERVAL}s until check $i/$CHECKS ..."
    sleep "$INTERVAL"

    ELAPSED=$(( $(date +%s) - START ))
    log "══════════════════════════════════════════"
    log "CHECK $i/$CHECKS  (elapsed: $((ELAPSED/3600))h $((ELAPSED%3600/60))m)"
    log "══════════════════════════════════════════"

    # ── GPU 0 ──
    if check_alive "$GPU0_PID"; then
        log "  GPU 0 PID=$GPU0_PID ($GPU0_LABEL): 🔄 RUNNING"
    else
        log "  GPU 0 PID=$GPU0_PID ($GPU0_LABEL): ✅ FINISHED"
        # Find the log file for the current GPU 0 experiment
        if [[ "$GPU0_LABEL" == *"all"* ]]; then
            check_errors "experiments/gpu0_all.log" || true
        else
            part_num=$(echo "$GPU0_LABEL" | grep -oP '\d+')
            check_errors "experiments/gpu0_part${part_num}.log" || true
        fi
        launch_next 0
    fi

    # ── GPU 1 ──
    if check_alive "$GPU1_PID"; then
        log "  GPU 1 PID=$GPU1_PID ($GPU1_LABEL): 🔄 RUNNING"
    else
        log "  GPU 1 PID=$GPU1_PID ($GPU1_LABEL): ✅ FINISHED"
        if [[ "$GPU1_LABEL" == *"part 3"* ]]; then
            check_errors "experiments/gpu1_part3.log" || true
        else
            part_num=$(echo "$GPU1_LABEL" | grep -oP '\d+')
            check_errors "experiments/gpu1_part${part_num}.log" || true
        fi
        launch_next 1
    fi

    gather_status

    # If both GPUs are idle and queue is empty, we're done
    if ! check_alive "$GPU0_PID" 2>/dev/null && ! check_alive "$GPU1_PID" 2>/dev/null; then
        if [ "$QUEUE_IDX" -ge "${#QUEUE[@]}" ]; then
            log "🎉 All experiments complete and queue empty. Exiting monitor."
            break
        fi
    fi
done

log "Monitor finished after $CHECKS checks."
echo "<augment-user-message>Experiment monitor completed after 48 hours. Check experiments/monitor.log for full history.</augment-user-message>"


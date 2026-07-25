#!/usr/bin/env bash
# Watch for a wave's completion sentinel and copy the wave directory to a
# persistent location before the pipeline's cleanup_intermediate logic can
# delete its model files.
#
# Usage:
#   persist_wave_to_data.sh <source_wave_dir> <dest_dir> [poll_seconds]
#
# Example:
#   nohup ./scripts/persist_wave_to_data.sh \
#       <project_root>/trainedmodels/full_mlp_pipeline_v2/wave6_layers_4_15 \
#       <pellm_data_root>/persisted_waves/full_mlp_pipeline_v2/wave6_layers_4_15 \
#       60 \
#       > <pellm_data_root>/persisted_waves/wave6_backup.log 2>&1 &
#
# Exit codes:
#   0 = backup succeeded (sizes match)
#   1 = sentinel never appeared (only if MAX_WAIT_HOURS exceeded)
#   2 = copy completed but size mismatch — manual verification required
#   3 = bad arguments

set -euo pipefail

SRC="${1:-}"
DST="${2:-}"
POLL_SECONDS="${3:-60}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-336}"  # 14 days, effectively unlimited

if [[ -z "$SRC" || -z "$DST" ]]; then
    echo "Usage: $0 <source_wave_dir> <dest_dir> [poll_seconds]" >&2
    exit 3
fi

SENTINEL="$SRC/wave_complete.json"
MAX_WAIT_SECONDS=$(( MAX_WAIT_HOURS * 3600 ))
WAITED=0

mkdir -p "$(dirname "$DST")"

echo "[$(date -Is)] watcher pid=$$ src=$SRC dst=$DST poll=${POLL_SECONDS}s"
echo "[$(date -Is)] waiting for sentinel: $SENTINEL"

while [[ ! -f "$SENTINEL" ]]; do
    if (( WAITED >= MAX_WAIT_SECONDS )); then
        echo "[$(date -Is)] FAIL: sentinel never appeared after ${MAX_WAIT_HOURS}h" >&2
        exit 1
    fi
    sleep "$POLL_SECONDS"
    WAITED=$(( WAITED + POLL_SECONDS ))
done

echo "[$(date -Is)] sentinel detected after ${WAITED}s of polling"

# Brief settle delay so any final file flushes complete before copy starts.
sleep 5

echo "[$(date -Is)] copying $SRC -> $DST"
mkdir -p "$DST"
# cp -a preserves mode/ownership/timestamps; trailing /. copies contents into DST.
cp -a "$SRC/." "$DST/"

if [[ ! -f "$DST/wave_complete.json" ]]; then
    echo "[$(date -Is)] FAIL: wave_complete.json missing at destination" >&2
    exit 2
fi

SIZE_SRC=$(du -sb "$SRC" | awk '{print $1}')
SIZE_DST=$(du -sb "$DST" | awk '{print $1}')
echo "[$(date -Is)] src bytes=$SIZE_SRC  dst bytes=$SIZE_DST"

if [[ "$SIZE_SRC" == "$SIZE_DST" ]]; then
    echo "[$(date -Is)] SUCCESS: wave persisted to $DST"
    exit 0
fi

echo "[$(date -Is)] WARNING: size mismatch (src=$SIZE_SRC dst=$SIZE_DST) — verify manually" >&2
exit 2

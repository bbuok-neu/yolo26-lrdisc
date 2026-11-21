#!/usr/bin/env bash
# Run after training completes (GPU idle for accurate FPS).
# Usage: BASE=baseline_full TEMP=temporal_full bash scripts/final_eval.sh
set -e
cd "$(dirname "$0")/.."
PY="C:/Users/bbuok/.conda/envs/vocod-yolo/python.exe"
BASE="${BASE:-baseline_full}"
TEMP="${TEMP:-temporal_full}"
echo "=== baseline streaming test ($BASE) ==="
"$PY" scripts/stream_eval.py --weights runs/$BASE/weights/best.pt --split test --out runs/stream_test_$BASE.json
echo "=== temporal streaming test ($TEMP) ==="
"$PY" scripts/stream_eval.py --weights runs/$TEMP/weights/best.pt --temporal --split test --out runs/stream_test_$TEMP.json
echo "=== temporal diagnostic (TemporalLens) ==="
"$PY" scripts/diagnose.py --weights runs/$TEMP/weights/best.pt --split test --max-frames 800 --out runs/diagnose_$TEMP.json
echo "=== qualitative visualization ==="
"$PY" scripts/visualize.py --baseline runs/$BASE/weights/best.pt --temporal runs/$TEMP/weights/best.pt --split test --n 6
echo "=== report ==="
"$PY" scripts/make_report.py --base "$BASE" --temp "$TEMP"
echo "ALL DONE"
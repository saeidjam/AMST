#!/usr/bin/env bash
# Steps after human annotation: calibration, second judge, statistics, tables, figures.
set -euo pipefail
OUT=${1:-results}
python -m amst calibrate --out "$OUT"
python -m amst judge     --out "$OUT"
python -m amst analyze   --out "$OUT"

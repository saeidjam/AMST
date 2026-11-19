#!/usr/bin/env bash
# Full AMST run. Requires OPENROUTER_API_KEY. Re-running is safe: every LLM call is cached.
set -euo pipefail
OUT=${1:-results}

python -m amst select   --out "$OUT"
python -m amst prompts  --out "$OUT"
python -m amst converse --out "$OUT"
python -m amst score    --out "$OUT"
python -m amst sample   --out "$OUT"

echo ""
echo "Now annotate:"
echo "  $OUT/annotation/prompt_sample_TO_ANNOTATE.csv   -> save as prompt_sample_ANNOTATED.csv"
echo "  $OUT/annotation/response_sample_TO_ANNOTATE.csv -> save as response_sample_ANNOTATED.csv"
echo "Then run:  bash scripts/finish.sh $OUT"

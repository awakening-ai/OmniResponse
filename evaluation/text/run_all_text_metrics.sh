#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 GEN_DIR REF_DIR" >&2
  exit 1
fi

GEN_DIR="$1"
REF_DIR="$2"

# Resolve directory of this script so it can be called from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$GEN_DIR" ]; then
  echo "GEN_DIR does not exist or is not a directory: $GEN_DIR" >&2
  exit 1
fi

if [ ! -d "$REF_DIR" ]; then
  echo "REF_DIR does not exist or is not a directory: $REF_DIR" >&2
  exit 1
fi

# -------------------------
# METEOR
# -------------------------
METEOR_OUTPUT="$(python3 "$SCRIPT_DIR/compute_meteor.py" \
  --gen_dir "$GEN_DIR" \
  --ref_dir "$REF_DIR")"
METEOR_SCORE="$(printf '%s\n' "$METEOR_OUTPUT" | grep 'Average METEOR score' | awk '{print $NF}')"

# -------------------------
# ROUGE-L
# -------------------------
ROUGEL_OUTPUT="$(python3 "$SCRIPT_DIR/compute_rouge-l.py" \
  --gen_dir "$GEN_DIR" \
  --ref_dir "$REF_DIR")"
ROUGEL_SCORE="$(printf '%s\n' "$ROUGEL_OUTPUT" | grep 'Average ROUGE-L' | awk '{print $NF}')"

# -------------------------
# BERTScore (P/R/F1 + #pairs)
# -------------------------
BERT_OUTPUT="$(python3 "$SCRIPT_DIR/compute_bertscore.py" \
  --gen_dir "$GEN_DIR" \
  --ref_dir "$REF_DIR")"

BERT_P="$(printf '%s\n' "$BERT_OUTPUT" | grep 'Average BERTScore-P'  | awk '{print $NF}')"
BERT_R="$(printf '%s\n' "$BERT_OUTPUT" | grep 'Average BERTScore-R'  | awk '{print $NF}')"
BERT_F1="$(printf '%s\n' "$BERT_OUTPUT" | grep 'Average BERTScore-F1' | awk '{print $NF}')"
NUM_PAIRS="$(printf '%s\n' "$BERT_OUTPUT" | grep '#Valid pairs'       | awk '{print $NF}')"

# -------------------------
# Distinct-1 / Distinct-2
# -------------------------
DISTINCT_OUTPUT="$(python3 "$SCRIPT_DIR/compute_distinct.py" \
  --gen_dir "$GEN_DIR")"

DISTINCT1="$(printf '%s\n' "$DISTINCT_OUTPUT" | grep 'Distinct-1' | awk '{print $NF}')"
DISTINCT2="$(printf '%s\n' "$DISTINCT_OUTPUT" | grep 'Distinct-2' | awk '{print $NF}')"

# -------------------------
# Final copy-friendly summary
# -------------------------

echo "MAIN  METEOR=${METEOR_SCORE}  BERTScoreF1=${BERT_F1}  ROUGE-L=${ROUGEL_SCORE}  Distinct-2=${DISTINCT2}"
echo "EXTRA Distinct-1=${DISTINCT1}  BERTScoreP=${BERT_P}  BERTScoreR=${BERT_R}  NumPairs=${NUM_PAIRS}"


#!/bin/bash
set -e

CONFIG_FILE="${1:-configs/pad_config.yaml}"
STUDENT_MODEL="${2:-Qwen/Qwen3-8B}"
SCORER_MODEL="${3:-Qwen/Qwen3-32B}"
DATASET="${4:-gsm8k}"
OUTPUT_DIR="${5:-./outputs}"

echo "================================================"
echo "PAD-Curriculum Training Launch"
echo "================================================"
echo "Config: $CONFIG_FILE"
echo "Student: $STUDENT_MODEL"
echo "Scorer: $SCORER_MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "================================================"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 pad_curriculum/main.py train \
    --config "$CONFIG_FILE" \
    --student-model "$STUDENT_MODEL" \
    --scorer-model "$SCORER_MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR"

echo "================================================"
echo "Training finished"
echo "================================================"

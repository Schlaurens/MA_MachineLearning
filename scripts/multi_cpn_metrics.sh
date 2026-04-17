#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2

LOG_DIR="logs/fit/cpn-grayscale"
MODEL_DIR="models/evaluation/cpn-grayscale"

for timestamp in "$LOG_DIR"/*/*; do
    timestamp=$(basename "$timestamp")
    uv run src/evaluation/cpn_metrics.py "$timestamp" "$LOG_DIR" "$MODEL_DIR"
done
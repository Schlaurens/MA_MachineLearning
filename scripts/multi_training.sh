#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2

SETTINGSFILES=(
    "all_categories_v1_240x320.yaml"
    "all_categories_v2_240x320.yaml"
    "all_categories_v3_240x320.yaml"
)

for F in "${SETTINGSFILES[@]}"; do
    echo "Running with settings file: $F"
    uv run src/training/train.py "$F"
done
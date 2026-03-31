#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2

# SETTINGSFILES=(
#     "all_categories_240x320_v1.yaml"
#     "all_categories_240x320_v2.yaml"
#     "all_categories_240x320_v3.yaml"
# )

# SETTINGSFILES=(
#     "all_categories_288x384_v1.yaml"
#     "all_categories_288x384_v2.yaml"
#     "all_categories_288x384_v3.yaml"
# )

# SETTINGSFILES=(
#     "all_categories_360x480_v1.yaml"
#     "all_categories_360x480_v2.yaml"
#     "all_categories_360x480_v3.yaml"
# )

SETTINGSFILES=(
    "all_categories_432x576_v1.yaml"
    "all_categories_432x576_v2.yaml"
    "all_categories_432x576_v3.yaml"
)

# SETTINGSFILES=(
#     "all_categories_480x640_v1.yaml"
#     "all_categories_480x640_v2.yaml"
#     "all_categories_480x640_v3.yaml"
# )

for F in "${SETTINGSFILES[@]}"; do
    echo "Running with settings file: $F"
    uv run src/training/train.py "$F"
done
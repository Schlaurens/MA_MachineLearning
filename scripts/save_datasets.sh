#!/bin/bash
# Suppress TensorFlow warnings and messages
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow logs (0 = all, 1 = info, 2 = warnings, 3 = errors)

# Check if the parent directory is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <parent_directory> <dest_directory> <height> <width>"
    exit 1
fi

parent_dir="$1"
dest_dir="$2"
height="$3"
width="$4"

# Check if the parent directory exists
if [ ! -d "$parent_dir" ]; then
    echo "Error: Directory '$parent_dir' does not exist."
    exit 1
fi

# Loop through each subdirectory
for dir in "$parent_dir"/*/; do
    if [ -d "$dir" ]; then
        echo "Generating .tfrecords file for directory: $dir"
        start_time=$(date +%s)
        uv run src/dataset/save_dataset.py --src_dir "$dir" --dest_dir "$dest_dir" --image_res $height $width
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "Time taken: $elapsed seconds"
        echo "----------------------------------------"
    fi
done
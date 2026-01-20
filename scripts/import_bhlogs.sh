#!/bin/bash

# Check if directories is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <parent_directory> <target_directory>"
    exit 1
fi

parent_dir="$1"
target_dir="$2"

# Check if the parent directory exists
if [ ! -d "$parent_dir" ]; then
    echo "Error: Source directory '$parent_dir' does not exist."
    exit 1
fi
if [ ! -d "$target_dir" ]; then
    echo "Error: Target directory '$target_dir' does not exist."
    exit 1
fi

for file in "$parent_dir"/*; do
    if [ -f "$file" ]; then
        echo "Extracting labels from $file into $target_dir"
        start_time=$(date +%s)
        uv run src/import_bhlogs.py "$file" --destination "$target_dir"
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "Time taken: $elapsed seconds"
        echo "----------------------------------------"
    fi
done
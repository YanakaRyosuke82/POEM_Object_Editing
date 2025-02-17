#!/bin/bash

# Check if directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

parent_dir="$1"

# Print header
echo "=== File Count Report ==="
echo "Parent directory: $parent_dir"
echo "========================"
echo

# Find all subdirectories and count files in each
find "$parent_dir" -type d | while read -r dir; do
    # Skip the parent directory itself
    if [ "$dir" = "$parent_dir" ]; then
        continue
    fi
    
    # Count files in current subdirectory
    file_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
    
    # Format output with proper indentation
    echo "Directory: ${dir#$parent_dir/}"
    echo "Files: $file_count"
    echo "------------------------"
done

# Print total count at the end
total_files=$(find "$parent_dir" -type f | wc -l)
echo "Total files in all directories: $total_files"

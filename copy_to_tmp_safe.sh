#!/bin/bash

file_path="$1"
echo "file_path: $file_path"
tmp_path="/tmp/$(basename "$file_path")"

if [ -z "$file_path" ]; then
    echo "Usage: $0 <file_path>"
    exit 1
fi

if [ -f "$tmp_path" ]; then
    echo "$tmp_path already exists. Waiting for 30 seconds..."
    sleep 30
else
    echo "Copying $file_path to $tmp_path"
    cp "$file_path" "$tmp_path"
fi

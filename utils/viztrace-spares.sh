#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <module>"
  exit 1
fi

# Assign the argument to a variable
module="$1"

# Generate the output file name by appending .json
output_file="${module}.json"

# Run viztracer with the generated output file and the command/script
viztracer -o "$output_file" --log_sparse -m "$module"

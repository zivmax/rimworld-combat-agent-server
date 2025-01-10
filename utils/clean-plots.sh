#!/bin/bash

# Find and remove all 'plots' directories in the project
find . -type d -name "plots" -exec rm -rf {} +

echo "All plot directories have been cleaned."
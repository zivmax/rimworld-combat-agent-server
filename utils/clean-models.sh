#!/bin/bash

# Find and remove all 'models' directories in the project
find . -type d -name "models" -exec rm -rf {} +

echo "All model directories have been cleaned."
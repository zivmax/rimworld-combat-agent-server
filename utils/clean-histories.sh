#!/bin/bash

# Find and remove all 'plots' directories in the project
find . -type d -name "histories" -exec rm -rf {} +

echo "All history directories have been cleaned."
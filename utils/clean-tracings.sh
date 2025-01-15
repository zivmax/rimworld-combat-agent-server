#!/bin/bash

# Find and remove all 'tracings' directories in the project
find . -type d -name "tracings" -exec rm -rf {} +

echo "All tracing directories have been cleaned."
#!/bin/bash

# Find and remove all 'logs' directories in the project
find . -type d -name "logs" -exec rm -rf {} +

echo "All log directories have been cleaned."
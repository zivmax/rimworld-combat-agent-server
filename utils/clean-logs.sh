#!/bin/bash

# Define the project root directory (current directory)
PROJECT_ROOT="/workspaces/agent-server"

# Find and remove all 'logs' directories in the project
find "$PROJECT_ROOT" -type d -name "logs" -exec rm -rf {} +

echo "All log directories have been cleaned."
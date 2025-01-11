#!/bin/bash

# Define the target file path
TARGET_FILE="/root/.config/unity3d/Ludeon Studios/RimWorld by Ludeon Studios/Config/ModsConfig.xml"

# Create the directory structure if it doesn't exist
mkdir -p "$(dirname "$TARGET_FILE")"

# Write the XML content to the target file
cat <<EOF > "$TARGET_FILE"
<?xml version="1.0" encoding="utf-8"?>
<ModsConfigData>
  <version>1.5.4243 rev889</version>
  <activeMods>
    <li>ludeon.rimworld</li>
    <li>zivmax.combatagent</li>
  </activeMods>
  <knownExpansions>
    <li>ludeon.rimworld.royalty</li>
    <li>ludeon.rimworld.ideology</li>
    <li>ludeon.rimworld.biotech</li>
    <li>ludeon.rimworld.anomaly</li>
  </knownExpansions>
</ModsConfigData>
EOF

# Output success message
echo "XML file created successfully at $TARGET_FILE"

# Add the safe directory to git
git config --global --add safe.directory /workspaces/agent-server
git config --global --add safe.directory /mnt/game/Mods/agent-client

# Output success message
echo "Safe directories added to git"
 
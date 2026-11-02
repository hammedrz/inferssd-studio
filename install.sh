#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Add InferSSD Studio to the app grid (for this user only). Run again after moving this folder.
# Remove it with: bash install.sh --uninstall
set -eu
here=$(cd "$(dirname "$0")" && pwd)
id=io.github.hammedrz.InferSSDStudio
apps=${XDG_DATA_HOME:-$HOME/.local/share}/applications
icons=${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$apps/$id.desktop" "$icons/$id.svg"
    echo "removed InferSSD Studio from the app grid"
    exit 0
fi

mkdir -p "$apps" "$icons"
cp "$here/$id.svg" "$icons/$id.svg"
cat > "$apps/$id.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=InferSSD Studio
Comment=Chat with and serve mixture-of-experts models whose experts stay on the SSD
Exec=/usr/bin/python3 $here/inferssd_studio.py
Icon=$id
Terminal=false
Categories=Development;
Keywords=LLM;AI;chat;inferssd;llama;
StartupNotify=true
StartupWMClass=$id
EOF
command -v update-desktop-database >/dev/null && update-desktop-database "$apps" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "${icons%/scalable/apps}" 2>/dev/null || true
echo "installed: search for \"InferSSD Studio\" in your apps"

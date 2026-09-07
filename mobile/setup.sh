#!/usr/bin/env bash
set -e

# mobile/setup.sh - copies the root static/ web app into mobile/www and installs npm deps

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILE_DIR="$ROOT_DIR/mobile"
STATIC_DIR="$ROOT_DIR/static"
WWW_DIR="$MOBILE_DIR/www"

echo "Setting up mobile folder..."

# Create mobile/www and copy static files
rm -rf "$WWW_DIR"
mkdir -p "$WWW_DIR"
cp -r "$STATIC_DIR/"* "$WWW_DIR/"

echo "Copied web assets from $STATIC_DIR to $WWW_DIR"

# Install npm deps
cd "$MOBILE_DIR"
if [ -f package.json ]; then
  echo "Installing npm dependencies in mobile/"
  npm ci || npm install
else
  echo "No package.json found in mobile/"
fi

echo "Setup complete. Run 'npx cap add android' (and 'npx cap add ios' on macOS) then 'npx cap copy' and 'npx cap open android' to build."

#!/bin/bash
# build-flatpak.sh — builds baEraser as a proper Flatpak
# OpenCV is built from source inside the flatpak sandbox.
# ONNX Runtime pre-built binary is bundled (glibc-compatible).
# Output: baEraser.flatpak bundle in the same directory as this script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_ID=si.generacija.baEraser
MANIFEST="$SCRIPT_DIR/si.generacija.baEraser.yml"
BUILD_DIR="$SCRIPT_DIR/_flatpak_build"
REPO_DIR="$SCRIPT_DIR/_flatpak_repo"
BUNDLE="$SCRIPT_DIR/baEraser.flatpak"

echo "=== baEraser Flatpak builder ==="
echo ""

# ── 1. Check flatpak-builder ─────────────────────────────────────────────────
FLATPAK_BUILDER=""
if command -v flatpak-builder &>/dev/null; then
    FLATPAK_BUILDER="flatpak-builder"
elif flatpak run org.flatpak.Builder --version &>/dev/null 2>&1; then
    FLATPAK_BUILDER="flatpak run org.flatpak.Builder"
else
    echo "flatpak-builder not found. Installing via eopkg..."
    sudo eopkg install -y flatpak-builder
    FLATPAK_BUILDER="flatpak-builder"
fi
echo "  flatpak-builder: $FLATPAK_BUILDER"

# ── 2. Check GNOME runtime ───────────────────────────────────────────────────
RUNTIME_VERSION=$(grep 'runtime-version' "$MANIFEST" | awk -F"'" '{print $2}')
echo "  Runtime: org.gnome.Platform//$RUNTIME_VERSION"

if ! flatpak info "org.gnome.Platform//$RUNTIME_VERSION" &>/dev/null; then
    echo "  Installing org.gnome.Platform//$RUNTIME_VERSION ..."
    flatpak install -y flathub "org.gnome.Platform//$RUNTIME_VERSION"
fi
if ! flatpak info "org.gnome.Sdk//$RUNTIME_VERSION" &>/dev/null; then
    echo "  Installing org.gnome.Sdk//$RUNTIME_VERSION ..."
    flatpak install -y flathub "org.gnome.Sdk//$RUNTIME_VERSION"
fi

# ── 3. Build with flatpak-builder ────────────────────────────────────────────
echo ""
echo "[1/3] Running flatpak-builder (OpenCV build may take 10-20 min first time)..."
rm -rf "$BUILD_DIR" "$REPO_DIR"

$FLATPAK_BUILDER \
    --repo="$REPO_DIR" \
    --force-clean \
    "$BUILD_DIR" \
    "$MANIFEST" \
    2>&1

# ── 4. Export bundle ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] Exporting .flatpak bundle..."
rm -f "$BUNDLE"
flatpak build-bundle \
    "$REPO_DIR" \
    "$BUNDLE" \
    "$APP_ID" \
    2>&1

SIZE=$(du -sh "$BUNDLE" | cut -f1)
echo ""
echo "  Done: $BUNDLE ($SIZE)"

# ── 5. Install locally ───────────────────────────────────────────────────────
echo ""
echo "[3/3] Installing locally..."
flatpak install --user -y --bundle "$BUNDLE" 2>&1 || true

echo ""
echo "  Run with:  flatpak run $APP_ID"
echo "  Uninstall: flatpak uninstall --user $APP_ID"
echo ""

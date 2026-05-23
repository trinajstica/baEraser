#!/bin/bash
# build-appimage.sh — builds baEraser.AppImage in the same directory as this script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP=baEraser
APPDIR="$SCRIPT_DIR/AppDir"
OUTPUT="$SCRIPT_DIR/${APP}.AppImage"
APPIMAGETOOL="${APPIMAGETOOL:-$(which appimagetool 2>/dev/null || echo /home/barko/bin/appimagetool)}"

# ── 1. Check tools ────────────────────────────────────────────────────────────
echo "=== baEraser AppImage builder ==="
echo ""

if [ ! -x "$APPIMAGETOOL" ]; then
    echo "ERROR: appimagetool not found at $APPIMAGETOOL"
    echo "  Download from: https://github.com/AppImage/AppImageKit/releases"
    exit 1
fi

for dep in pkg-config g++ make; do
    if ! command -v "$dep" &>/dev/null; then
        echo "ERROR: missing tool: $dep"
        exit 1
    fi
done

# ── 2. Build binary ───────────────────────────────────────────────────────────
echo "[1/5] Building binary..."
make -j"$(nproc)" all
echo "      OK: build/$APP"

# ── 3. Prepare AppDir ─────────────────────────────────────────────────────────
echo "[2/5] Preparing AppDir..."
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/lib"
mkdir -p "$APPDIR/usr/share/icons/hicolor/scalable/apps"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/models"

# Binary
cp "build/$APP" "$APPDIR/usr/bin/$APP"

# Desktop file & icon
cp "data/baEraser.desktop"         "$APPDIR/usr/share/applications/baEraser.desktop"
cp "data/baEraser.desktop"         "$APPDIR/$APP.desktop"
cp "data/icons/baEraser.svg"       "$APPDIR/usr/share/icons/hicolor/scalable/apps/baEraser.svg"
cp "data/icons/baEraser.svg"       "$APPDIR/baEraser.svg"

# LaMa models (optional — only copy if present)
for model in models/*.onnx; do
    [ -f "$model" ] && cp "$model" "$APPDIR/models/" && echo "      Bundled: $model"
done

# ── 4. Bundle libraries ───────────────────────────────────────────────────────
echo "[3/5] Bundling libraries..."

bundle_lib() {
    local lib="$1"
    local dest="$APPDIR/usr/lib"
    if [ -f "$lib" ] && [ ! -f "$dest/$(basename "$lib")" ]; then
        cp -L "$lib" "$dest/"
        echo "      + $(basename "$lib")"
    fi
}

# OpenCV libs (bundle — not commonly available on all distros)
for lib in $(ldd "build/$APP" | awk '/libopencv/{print $3}'); do
    bundle_lib "$lib"
done

# ONNX Runtime (bundled in third_party — always copy)
ORT_LIB_DIR="$SCRIPT_DIR/third_party/onnxruntime-linux-x64-1.26.0/lib"
for lib in "$ORT_LIB_DIR"/libonnxruntime*.so*; do
    [ -f "$lib" ] && bundle_lib "$lib"
done

# GTK4, libadwaita, glib — intentionally NOT bundled:
# these must come from the host system to get correct themes, renderers, and
# platform integration. Bundling them causes more problems than it solves.

# ── 5. AppRun wrapper ─────────────────────────────────────────────────────────
echo "[4/5] Writing AppRun..."
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"

# Prepend bundled libs so OpenCV / ORT are found
export LD_LIBRARY_PATH="$HERE/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Let the app find models relative to AppImage location
export BAERASER_MODEL_PATH="$HERE/models"

exec "$HERE/usr/bin/baEraser" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# ── 6. Build AppImage ─────────────────────────────────────────────────────────
echo "[5/5] Packing AppImage..."
rm -f "$OUTPUT" "$SCRIPT_DIR"/baEraser-*-x86_64.AppImage

ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$OUTPUT" 2>&1

if [ -f "$OUTPUT" ]; then
    SIZE=$(du -sh "$OUTPUT" | cut -f1)
    echo ""
    echo "  Done: $OUTPUT ($SIZE)"
    echo ""
    echo "  Run with:  $OUTPUT"
else
    echo "ERROR: appimagetool did not produce output file"
    exit 1
fi

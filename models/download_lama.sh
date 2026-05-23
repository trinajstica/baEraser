#!/bin/bash
# Prenese uradni LaMa ONNX model (OpenCV zoo / Carve/LaMa-ONNX)
set -e

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$MODELS_DIR/inpainting_lama_2025jan.onnx"

if [ -f "$OUT" ]; then
    echo "Model že obstaja: $OUT"
    exit 0
fi

echo "=== Prenos LaMa ONNX modela ==="
echo "Vir: HuggingFace – opencv/inpainting_lama"
echo ""

# Primary: OpenCV zoo official model (newest, ~100MB)
URL_OPENCV="https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx"
# Fallback: Carve/LaMa-ONNX fp32 (~100MB)
URL_CARVE="https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

download() {
    local url="$1"
    local out="$2"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$out" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$out" "$url"
    else
        echo "NAPAKA: wget ali curl nista nameščena."
        exit 1
    fi
}

echo "Prenašam iz OpenCV zoo..."
if download "$URL_OPENCV" "$OUT" 2>/dev/null; then
    echo ""
    echo "Shranjeno: $OUT"
    echo "Zaženi aplikacijo in izberi 'LaMa AI' metodo."
else
    echo "OpenCV zoo ni dostopen, preskušam Carve/LaMa-ONNX..."
    FALLBACK="$MODELS_DIR/lama_fp32.onnx"
    if download "$URL_CARVE" "$FALLBACK" 2>/dev/null; then
        echo ""
        echo "Shranjeno: $FALLBACK"
        echo "Zaženi aplikacijo in izberi 'LaMa AI' metodo."
    else
        echo "NAPAKA: Prenos ni uspel. Preveri internetno povezavo."
        echo ""
        echo "Ročni prenos:"
        echo "  $URL_OPENCV"
        echo "  → shrani kot: $OUT"
        exit 1
    fi
fi

#!/bin/bash
# baEraser – complete local build script
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

ORT_VERSION="1.26.0"
ORT_DIR="$PROJECT_DIR/third_party/onnxruntime-linux-x64-$ORT_VERSION"
ORT_TARBALL="onnxruntime-linux-x64-$ORT_VERSION.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_TARBALL"

MODEL_DIR="$PROJECT_DIR/models"
PRIMARY_MODEL="$MODEL_DIR/inpainting_lama_2025jan.onnx"
FALLBACK_MODEL="$MODEL_DIR/lama_fp32.onnx"
CARVE_MODEL="$MODEL_DIR/lama_carve_fp32.onnx"

echo "=== baEraser build ==="
echo ""

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

run_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif have_cmd sudo; then
        sudo "$@"
    else
        echo "NAPAKA: za namestitev paketov potrebujem sudo ali root uporabnika."
        exit 1
    fi
}

download_file() {
    local url="$1"
    local out="$2"

    if have_cmd wget; then
        wget -q --show-progress -O "$out" "$url"
    elif have_cmd curl; then
        curl -L --progress-bar -o "$out" "$url"
    else
        install_system_dependencies "download-tools"
        if have_cmd wget; then
            wget -q --show-progress -O "$out" "$url"
        elif have_cmd curl; then
            curl -L --progress-bar -o "$out" "$url"
        else
            echo "NAPAKA: wget ali curl nista na voljo."
            exit 1
        fi
    fi
}

pkg_config_missing() {
    local missing=0
    for dep in "$@"; do
        if ! pkg-config --exists "$dep" 2>/dev/null; then
            missing=1
            echo "  manjka pkg-config modul: $dep"
        fi
    done
    return "$missing"
}

install_system_dependencies() {
    local mode="${1:-all}"

    echo ""
    echo "[1/4] Preverjam sistemske odvisnosti..."

    if [ "$mode" = "all" ] && pkg_config_missing gtk4 libadwaita-1 opencv4; then
        echo "  OK: GTK4, libadwaita in OpenCV so na voljo."
        return 0
    fi

    echo "  Poskus namestitve manjkajocih paketov..."

    if have_cmd eopkg; then
        if [ "$mode" = "all" ]; then
            run_root eopkg install -y -c system.devel
            run_root eopkg install -y pkgconf gtk4-devel libadwaita-devel opencv-devel wget curl tar
        else
            run_root eopkg install -y wget curl tar
        fi
    elif have_cmd apt-get; then
        run_root apt-get update
        if [ "$mode" = "all" ]; then
            run_root apt-get install -y build-essential pkg-config libgtk-4-dev libadwaita-1-dev libopencv-dev wget curl tar
        else
            run_root apt-get install -y wget curl tar
        fi
    elif have_cmd dnf; then
        if [ "$mode" = "all" ]; then
            run_root dnf install -y gcc-c++ make pkgconf-pkg-config gtk4-devel libadwaita-devel opencv-devel wget curl tar
        else
            run_root dnf install -y wget curl tar
        fi
    elif have_cmd pacman; then
        if [ "$mode" = "all" ]; then
            run_root pacman -S --needed --noconfirm base-devel pkgconf gtk4 libadwaita opencv wget curl tar
        else
            run_root pacman -S --needed --noconfirm wget curl tar
        fi
    elif have_cmd zypper; then
        if [ "$mode" = "all" ]; then
            run_root zypper install -y gcc-c++ make pkg-config gtk4-devel libadwaita-devel opencv-devel wget curl tar
        else
            run_root zypper install -y wget curl tar
        fi
    else
        echo "NAPAKA: ne poznam package managerja na tem sistemu."
        echo "Namesti rocno: g++, make, pkg-config, GTK4 dev, libadwaita dev, OpenCV dev, wget ali curl."
        exit 1
    fi

    if [ "$mode" = "all" ]; then
        echo "  Ponovno preverjam pkg-config module..."
        if ! pkg_config_missing gtk4 libadwaita-1 opencv4; then
            echo "NAPAKA: po namestitvi se vedno manjkajo build odvisnosti."
            exit 1
        fi
    fi
}

ensure_onnxruntime() {
    echo ""
    echo "[2/4] Preverjam ONNX Runtime..."

    if [ -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ] && [ -f "$ORT_DIR/lib/libonnxruntime.so.1.26.0" ]; then
        echo "  OK: $ORT_DIR"
        return 0
    fi

    echo "  Prenasam ONNX Runtime $ORT_VERSION..."
    mkdir -p "$PROJECT_DIR/third_party"

    local tmp_dir
    tmp_dir="$(mktemp -d)"

    download_file "$ORT_URL" "$tmp_dir/$ORT_TARBALL"
    tar -xzf "$tmp_dir/$ORT_TARBALL" -C "$PROJECT_DIR/third_party"
    rm -rf "$tmp_dir"

    if [ ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ] || [ ! -f "$ORT_DIR/lib/libonnxruntime.so.1.26.0" ]; then
        echo "NAPAKA: ONNX Runtime prenos ali razpakiranje ni uspelo."
        exit 1
    fi

    echo "  OK: ONNX Runtime je pripravljen."
}

ensure_lama_model() {
    echo ""
    echo "[3/4] Preverjam LaMa modele..."

    if [ -f "$PRIMARY_MODEL" ] || [ -f "$CARVE_MODEL" ] || [ -f "$FALLBACK_MODEL" ]; then
        echo "  OK: najden je vsaj en LaMa ONNX model."
        return 0
    fi

    echo "  LaMa model ni najden. Prenašam privzeti model..."
    mkdir -p "$MODEL_DIR"
    bash "$MODEL_DIR/download_lama.sh"

    if [ ! -f "$PRIMARY_MODEL" ] && [ ! -f "$CARVE_MODEL" ] && [ ! -f "$FALLBACK_MODEL" ]; then
        echo "NAPAKA: LaMa model ni bil prenesen."
        exit 1
    fi

    echo "  OK: LaMa model je pripravljen."
}

build_project() {
    echo ""
    echo "[4/4] Gradim projekt..."
    make -j"$(nproc)" all
}

install_system_dependencies "all"
ensure_onnxruntime
ensure_lama_model
build_project

echo ""
echo "=== Koncano ==="
echo "Binarna datoteka: $PROJECT_DIR/build/baEraser"
echo ""
echo "Zagon:"
echo "  ./build/baEraser"
echo "  make run"
echo ""

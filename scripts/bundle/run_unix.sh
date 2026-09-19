#!/usr/bin/env bash
# 2D Thrombosis Surrogate Predict -- macOS / Linux launcher.
#
# Shipped in the macos-linux release zip twice: as `run.command` (double-clickable in the macOS
# Finder) and as `run.sh`. The Windows zip carries its own embedded Python; this one cannot, so
# the first launch builds a private environment in `.venv/` next to this file from the exact
# package versions the release was tested with (`requirements-lock.txt`), and later launches
# reuse it.
#
# Supported: Apple Silicon Macs and x86-64 Linux, with Python 3.12-3.14. Intel Macs are not --
# PyTorch no longer publishes Intel-Mac builds -- and neither is ARM Linux, which gmsh (the
# mesher) does not build for.
set -euo pipefail
cd "$(dirname "$0")"

pause_and_exit() {
    echo
    read -r -p "Press Enter to close this window." _ || true
    exit "${1:-1}"
}
fail() {
    echo
    echo "ERROR: $*"
    pause_and_exit 1
}
trap 'fail "setup stopped unexpectedly (see the messages above)."' ERR

echo "2D Thrombosis Surrogate Predict"
echo "==============================="

os="$(uname -s)"
arch="$(uname -m)"
if [ "$os" = "Darwin" ] && [ "$arch" = "x86_64" ] \
        && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
    arch="arm64"   # an Apple Silicon Mac running this Terminal under Rosetta
fi
case "$os/$arch" in
    Darwin/arm64 | Linux/x86_64) ;;
    Darwin/x86_64) fail "Intel Macs are not supported: PyTorch no longer publishes Intel-Mac builds. An Apple Silicon (M-series) Mac is required." ;;
    *) fail "unsupported platform $os/$arch. Supported: Apple Silicon Macs and x86-64 Linux (or the Windows zip)." ;;
esac

# --- Find a usable Python --------------------------------------------------------------------
PY=""
for cand in python3.13 python3.12 python3.14 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    ok="$("$cand" -c 'import sys, platform; v = sys.version_info[:2]; print(int((3, 12) <= v <= (3, 14)), platform.machine())' 2>/dev/null || true)"
    case "$ok" in
        "1 arm64" | "1 x86_64")
            if [ "$os" = "Darwin" ] && [ "${ok#1 }" != "arm64" ]; then continue; fi
            PY="$(command -v "$cand")"; break ;;
    esac
done
if [ -z "$PY" ]; then
    echo "This app needs Python 3.12, 3.13 or 3.14 installed on this computer."
    if [ "$os" = "Darwin" ]; then
        echo "Install Python 3.13 (\"macOS 64-bit universal2 installer\") from"
        echo "    https://www.python.org/downloads/"
        echo "then double-click run.command again."
    else
        echo "Install it with your package manager, e.g.  sudo apt install python3.12-venv"
        echo "then run ./run.sh again."
    fi
    pause_and_exit 1
fi
echo "[i] Using $("$PY" --version) at $PY"

# --- One-time environment setup ----------------------------------------------------------------
VENV=".venv"
STAMP="$VENV/.installed-from"
LOCK_ID="$("$PY" -c 'import hashlib, sys; print(hashlib.sha256(open("requirements-lock.txt", "rb").read()).hexdigest()[:16])')"
if [ ! -x "$VENV/bin/python" ] || [ "$(cat "$STAMP" 2>/dev/null || true)" != "$LOCK_ID" ]; then
    echo "[i] First launch: setting up the app's own Python environment in $VENV/"
    echo "    This downloads about 1 GB of packages and takes several minutes. It only happens once."
    rm -rf "$VENV"
    "$PY" -m venv "$VENV" || fail "could not create a virtual environment. On Linux, install the venv module (e.g. sudo apt install python3.12-venv)."
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    if [ "$os" = "Linux" ]; then
        # PyPI's Linux torch is the multi-GB CUDA build; the app runs on CPU, so take the CPU wheel.
        torch_pin="$(grep -i '^torch==' requirements-lock.txt)"
        "$VENV/bin/python" -m pip install "$torch_pin" --index-url https://download.pytorch.org/whl/cpu
    fi
    "$VENV/bin/python" -m pip install -r requirements-lock.txt
    echo "$LOCK_ID" > "$STAMP"
    echo "[OK] Environment ready."
fi

# gmsh's Linux wheel links against system graphics/OpenMP libraries that minimal installs lack
# (measured on Ubuntu 24.04 under WSL: libGLU, libOpenGL, libXft, libgomp). Say which package to install
# rather than letting the app die on a ctypes traceback.
if ! gmsh_err="$("$VENV/bin/python" -c 'import gmsh' 2>&1)"; then
    echo "$gmsh_err" | tail -n 1
    echo
    echo "The mesher (gmsh) needs a few system libraries this computer does not have."
    if [ "$os" = "Linux" ]; then
        echo "Install them, then run ./run.sh again:"
        echo "    Debian / Ubuntu:  sudo apt install libglu1-mesa libopengl0 libxft2 libgomp1"
        echo "    Fedora / RHEL:    sudo dnf install mesa-libGLU libglvnd-opengl libXft libgomp"
    fi
    pause_and_exit 1
fi

trap - ERR
echo
echo "Starting the app. A browser tab opens once it is ready."
echo "Leave this window open while you use the app; close it (or press Ctrl+C) to stop."
echo
"$VENV/bin/python" -m src.tools.customer_predict_web --cpu || pause_and_exit $?

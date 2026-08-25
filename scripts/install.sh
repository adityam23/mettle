#!/bin/sh
# mettle installer: fetches the latest GitHub release and installs the
# `mettle` CLI and `mettled` daemon into a user prefix (default ~/.local/bin).
#
# Env overrides:
#   METTLE_INSTALL_DIR  binary target dir   (default ~/.local/bin)
#   METTLE_VARIANT      "cuda" selects the CUDA-enabled Linux build
#   METTLE_REPO         source repository   (default adityam23/mettle)
set -eu

REPO="${METTLE_REPO:-adityam23/mettle}"
INSTALL_DIR="${METTLE_INSTALL_DIR:-$HOME/.local/bin}"
VARIANT="${METTLE_VARIANT:-}"

os=$(uname -s)
arch=$(uname -m)
case "$os:$arch" in
    Linux:x86_64)
        asset="mettle-linux-amd64"
        [ "$VARIANT" = "cuda" ] && asset="mettle-linux-amd64-cuda"
        ;;
    Darwin:arm64) asset="mettle-macos-arm64" ;;
    *)
        echo "mettle-install: unsupported platform ${os}/${arch}" >&2
        echo "(source build: cargo build --release)" >&2
        exit 1
        ;;
esac

echo "mettle-install: resolving latest release of ${REPO} (${asset})"
url=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" |
    grep -o "\"browser_download_url\": *\"[^\"]*${asset}\.tar\.gz\"" |
    grep -o 'https://[^"]*') || {
    echo "mettle-install: no release asset found for ${asset}" >&2
    exit 1
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "mettle-install: downloading ${url}"
curl -fL "$url" -o "$tmp/archive.tar.gz"
tar xzf "$tmp/archive.tar.gz" -C "$tmp"

mkdir -p "$INSTALL_DIR"
for bin in mettle mettled; do
    if [ -f "$tmp"/*/"$bin" ]; then
        mv "$tmp"/*/"$bin" "$INSTALL_DIR/$bin"
        chmod +x "$INSTALL_DIR/$bin"
    fi
done

# Archives may bundle non-base shared libraries (CUDA runtime, OpenBLAS).
if [ -d "$tmp"/*/lib ]; then
    libdir="${METTLE_LIB_DIR:-$HOME/.local/lib/mettle}"
    mkdir -p "$libdir"
    cp "$tmp"/*/lib/*.so* "$libdir/" 2>/dev/null || true
    cat <<EOF

Bundled shared libraries installed to ${libdir}.
Add them to the loader path before starting the daemon:

    export LD_LIBRARY_PATH="${libdir}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"

EOF
fi

# XDG directories used by mettled on first run.
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/mettle"
mkdir -p "${XDG_DATA_HOME:-$HOME/.local/share}/mettle"

case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
        cat <<EOF

NOTE: ${INSTALL_DIR} is not on your PATH. Add to your shell profile:

    export PATH="${INSTALL_DIR}:\$PATH"

EOF
        ;;
esac

echo "mettle-install: done. Try:"
echo "  mettle tune                 # detect GPU, write tuned config"
echo "  mettle serve                # start mettled in the foreground"

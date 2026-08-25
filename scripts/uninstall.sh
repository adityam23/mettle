#!/bin/sh
# mettle uninstaller: removes the `mettle` and `mettled` binaries.
#
#   uninstall.sh          remove binaries only
#   uninstall.sh --purge  also delete config, state and CUDA libs
set -eu

INSTALL_DIR="${METTLE_INSTALL_DIR:-$HOME/.local/bin}"
LIB_DIR="${METTLE_LIB_DIR:-$HOME/.local/lib/mettle}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/mettle"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/mettle"

rm -f "$INSTALL_DIR/mettle" "$INSTALL_DIR/mettled"
echo "mettle-uninstall: removed binaries from $INSTALL_DIR"

if [ "${1:-}" = "--purge" ]; then
    rm -rf "$CONFIG_DIR" "$DATA_DIR" "$LIB_DIR"
    echo "mettle-uninstall: purged $CONFIG_DIR $DATA_DIR $LIB_DIR"
else
    echo "mettle-uninstall: kept config/state ($CONFIG_DIR, $DATA_DIR)."
    echo "Run with --purge to delete them."
fi

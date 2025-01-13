#!/bin/bash

set -e

# Wow, trying to avoid rebuilds with pyo3/maturin really sucks. See
# https://github.com/PyO3/PyO3/issues/1708

export script_dir=$(dirname $(realpath $0))
export PYO3_PYTHON=$script_dir/venv/bin/python
export PYO3_ENVIRONMENT_SIGNATURE=cpython-3.13-64bit
export CARGO_TARGET_DIR=maturin_target
#export CARGO_LOG=cargo::core::compiler::fingerprint=info

cargo run --bin=stub_gen "$@"
maturin develop "$@"

# Help gdb find pretty printers. Should be automatic, but is not, see https://github.com/rust-lang/rust/issues/96365
SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
OUT_FILE=$SITE_PACKAGES/palace/palace.*.so
objcopy --add-section .debug_gdb_scripts=<(echo -ne "\x01gdb_load_rust_pretty_printers.py\0") $OUT_FILE

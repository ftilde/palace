#!/bin/sh
#export PYO3_PYTHON=$(realpath ./venv/bin/python)
#export PYO3_ENVIRONMENT_SIGNATURE=cpython-3.12-64bit

cargo run --bin=stub_gen "$@"
CARGO_TARGET_DIR=maturin_target maturin develop "$@"

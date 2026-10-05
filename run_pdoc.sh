#!/bin/bash
set -o errexit

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "${PROJECT_PATH}"

rm -rf docs

# Requires pdoc: pip3 install pdoc
pdoc -o docs "wavenet_coding"

popd

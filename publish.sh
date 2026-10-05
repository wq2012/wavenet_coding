#!/bin/bash
set -o errexit

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "${PROJECT_PATH}"

# Clean up previous builds
rm -rf build
rm -rf dist
rm -rf *.egg-info

# Build package
python3 setup.py sdist bdist_wheel

# Upload to PyPI
python3 -m twine upload dist/*

popd

#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add project modules to PYTHONPATH.
if [[ "${PYTHONPATH}" != *"${PROJECT_PATH}"* ]]; then
  export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
fi

pushd "${PROJECT_PATH}"

# Execute all unit tests with coverage tracking.
rm -f .coverage
for TEST_FILE in $(find tests -name "*_test.py" | sort); do
  echo "Running tests in ${TEST_FILE}"
  python3 -m coverage run --source=wavenet_coding -a "${TEST_FILE}"
done
python3 -m coverage report -m
echo "All tests passed!"

popd

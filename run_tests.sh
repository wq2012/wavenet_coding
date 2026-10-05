#!/bin/bash
set -o errexit

# Execute all unit tests with coverage tracking
rm -f .coverage
for test_file in tests/*_test.py; do
  coverage run --source=wavenet_coding -a "$test_file"
done
coverage report -m

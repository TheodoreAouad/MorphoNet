#!/bin/bash

set -e

# See https://github.com/pytorch/pytorch/issues/32277#issuecomment-588649041
export CC=gcc-8
export CXX=g++-8

python3.9 setup.py install --user

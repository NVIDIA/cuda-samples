#!/bin/sh

mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make && make test && make install

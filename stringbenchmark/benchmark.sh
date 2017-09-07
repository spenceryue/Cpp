#!/bin/bash

echo "(1) indexes"
time ./bin/index.exe > /dev/null

echo "(2) indexes over C string"
time ./bin/cindex.exe > /dev/null

echo "(3) string::at()"
time ./bin/stringat.exe > /dev/null

echo "(4) pointers over C string"
time ./bin/cpointers.exe > /dev/null

echo "(5) iterators"
time ./bin/iterators.exe > /dev/null

echo "(6) const_iterators"
time ./bin/const_iterators.exe > /dev/null

echo "(7) for_each"
time ./bin/for_each.exe > /dev/null

echo "(8) for_reference_range_loop"
time ./bin/for_reference_range_loop.exe > /dev/null

echo "(9) for_const_reference_range_loop"
time ./bin/for_const_reference_range_loop.exe > /dev/null
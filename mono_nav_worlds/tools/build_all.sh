#!/usr/bin/env bash
# Regenerate every world of the benchmark (same seeds -> identical worlds).
set -e
cd "$(dirname "$0")"
G="python3 generate_world.py"
# training worlds: varied subsets of the hard cases
$G --name train_01 --seed 1 --hard glass,blank
$G --name train_02 --seed 2 --hard blank,dark
$G --name train_03 --seed 3 --hard glass,dark
$G --name train_04 --seed 4 --hard glass,blank,dark
$G --name train_05 --seed 5
# test worlds: unseen layouts; never use them for training
$G --name test_clean --seed 101
$G --name test_01 --seed 102 --hard glass,blank,dark
$G --name test_02 --seed 103 --hard glass,blank,dark
$G --name test_03 --seed 104 --hard glass,blank,dark

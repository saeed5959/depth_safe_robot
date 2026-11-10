#!/usr/bin/env bash
# Generate the fixed episode files (run once; commit episodes/ with the paper code).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
WD=${WORLDS_DIR:-$HOME/depth_safe_robot/mono_nav_worlds}/worlds
cd "$HERE"
for w in test_clean test_01 test_02 test_03; do
  python3 gen_episodes.py --worlds_dir "$WD" --world $w --out episodes/$w.yaml
done
for w in train_01 train_02 train_03 train_04 train_05; do
  python3 gen_episodes.py --worlds_dir "$WD" --world $w --train --out episodes/$w.yaml
done

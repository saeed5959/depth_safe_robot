#!/usr/bin/env bash
# Downloads the AWS small-house furniture models (MIT-0) that the worlds use.
set -e
cd "$(dirname "$0")"
mkdir -p third_party
if [ ! -d third_party/aws-robomaker-small-house-world ]; then
  git clone --depth 1 -b ros2 \
    https://github.com/IntelligentRoboticsLabs/aws-robomaker-small-house-world.git \
    third_party/aws-robomaker-small-house-world
fi
echo "OK: models in $(pwd)/third_party/aws-robomaker-small-house-world/models"

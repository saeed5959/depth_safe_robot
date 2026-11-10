#!/usr/bin/env bash
# Run every (world, method) combination: start the stack, run the episodes, shut down.
#
#   ./run_batch.sh eval      # test worlds,  gt + mono       -> results/results.csv
#   ./run_batch.sh collect   # train worlds, gt + mono, records RGB+GT depth -> dataset/
#   ./run_batch.sh smoke     # test_02, gt, 3 episodes, with RViz  (validate the pipeline)
#
# Override with env vars, e.g.  WORLDS="test_01" METHODS="mono" RTF=0.5 ./run_batch.sh eval
# Needs: ROS 2 sourced, ros2_ws sourced (mono_nav), $DA set.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-eval}
WORLDS_DIR=${WORLDS_DIR:-$HOME/depth_safe_robot/mono_nav_worlds}
RTF=${RTF:-0}                     # e.g. 0.5 if mono_depth can't keep up in real time
EXTRA_LAUNCH=${EXTRA_LAUNCH:-}    # e.g. "gt_sync:=true input_size:=364"
RUNNER_ARGS=${RUNNER_ARGS:-}
ROS2_WS=${ROS2_WS:-$HOME/depth_safe_robot/ros2_ws}

# ---- environment checks (fail fast instead of waiting for Nav2 forever)
set +u
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
if [ -f "$ROS2_WS/install/setup.bash" ]; then
  source "$ROS2_WS/install/setup.bash"
else
  echo "ERROR: $ROS2_WS/install/setup.bash not found (set ROS2_WS=... or colcon build)"; exit 1
fi
set -u
if ! ros2 pkg prefix mono_nav >/dev/null 2>&1; then
  echo "ERROR: package mono_nav not found after sourcing $ROS2_WS"; exit 1
fi
for exe in mono_depth odom_to_tf; do
  ros2 pkg executables mono_nav | grep -q " $exe$" || { echo "ERROR: mono_nav has no '$exe' executable"; exit 1; }
done
if [ -z "${DA:-}" ] || [ ! -d "$DA" ]; then
  echo "ERROR: \$DA is not set to the Depth-Anything-V2 repo"; exit 1
fi
if [ ! -f "$WORLDS_DIR/launch/world.launch.py" ]; then
  echo "ERROR: $WORLDS_DIR/launch/world.launch.py not found (set WORLDS_DIR=...)"; exit 1
fi

case $MODE in
  eval)    WORLDS=${WORLDS:-"test_clean test_01 test_02 test_03"}; METHODS=${METHODS:-"gt mono"}
           OUT=${OUT:-$HERE/results}; COLLECT=false; RVIZ=false ;;
  collect) WORLDS=${WORLDS:-"train_01 train_02 train_03 train_04 train_05"}; METHODS=${METHODS:-"gt mono"}
           OUT=${OUT:-$HERE/results_train}; COLLECT=true; RVIZ=false ;;
  smoke)   WORLDS=${WORLDS:-"test_02"}; METHODS=${METHODS:-"gt"}
           OUT=${OUT:-$HERE/results_smoke}; COLLECT=false; RVIZ=true
           RUNNER_ARGS="$RUNNER_ARGS --limit 3" ;;
  *) echo "usage: $0 eval|collect|smoke"; exit 1 ;;
esac

cleanup() {
  [ -n "${PID:-}" ] && kill -INT -- -"$PID" 2>/dev/null
  sleep 5
  [ -n "${PID:-}" ] && kill -KILL -- -"$PID" 2>/dev/null
  pkill -9 -f "ign gazebo" 2>/dev/null; pkill -9 -f "gz sim" 2>/dev/null
  pkill -9 -f parameter_bridge 2>/dev/null; pkill -9 -f "nav2_" 2>/dev/null
  pkill -9 -f collect_data.py 2>/dev/null
  ros2 daemon stop >/dev/null 2>&1
  PID=""
  sleep 3
}
trap 'cleanup; exit 130' INT TERM

mkdir -p "$OUT/logs"
for w in $WORLDS; do
  EP=$HERE/episodes/$w.yaml
  if [ ! -f "$EP" ]; then echo "missing $EP (run gen_all.sh)"; continue; fi
  for m in $METHODS; do
    echo "================ $w / $m ================"
    cleanup
    setsid ros2 launch "$HERE/eval_stack.launch.py" world:=$w method:=$m worlds_dir:=$WORLDS_DIR \
        rviz:=$RVIZ collect:=$COLLECT dataset:=$HERE/dataset $EXTRA_LAUNCH \
        > "$OUT/logs/${w}_${m}_stack.log" 2>&1 &
    PID=$!
    sleep 20
    LOG="$OUT/logs/${w}_${m}_stack.log"
    if ! kill -0 "$PID" 2>/dev/null || grep -q "Caught exception in launch" "$LOG"; then
      echo "ERROR: stack failed to start for $w / $m. Last lines of $LOG:"
      tail -n 15 "$LOG"
      cleanup; exit 1
    fi
    python3 "$HERE/episode_runner.py" --episodes "$EP" --method $m --out "$OUT" \
        --worlds_dir "$WORLDS_DIR/worlds" --rtf $RTF $RUNNER_ARGS \
        2>&1 | tee "$OUT/logs/${w}_${m}_runner.log"
  done
done
cleanup
python3 "$HERE/analyze.py" "$OUT/results.csv" | tee "$OUT/summary.md"

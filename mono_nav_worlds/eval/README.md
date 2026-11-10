# mono_nav_eval — episodes, batch evaluation, dataset collection

Put this folder next to your worlds: `~/depth_safe_robot/mono_nav_worlds/eval/`.
The scripts run with your sourced ROS 2 Humble environment; no colcon build needed.

| File | What |
|---|---|
| `world_map.py` | Occupancy / clearance map from a world SDF (walls, boxes, glass, AWS furniture footprints) + A* |
| `gen_episodes.py`, `gen_all.sh` | Fixed, seeded start/goal episodes per world (+ preview PNG) |
| `episodes/*.yaml` | Already generated for all 9 worlds (test: 30 episodes each, train: 40) |
| `eval_stack.launch.py` | Your 7 terminals in one launch: `world:=… method:=gt|mono` |
| `episode_runner.py` | Teleport → goal → monitor → outcome, writes `results.csv` + trajectories |
| `collect_data.py` | Saves RGB + GT depth (16-bit mm PNG) + pose while an episode runs |
| `run_batch.sh` | Loops worlds × methods: `smoke`, `eval`, `collect` |
| `analyze.py` | SR with Wilson CI, outcome breakdown, SPL, per-world/difficulty/hard-case, McNemar |

## 0. One-time changes to your sim

### a) Bumper (contact sensor) for collision detection
In `mono_bot.urdf`, add inside `<robot>`:
```xml
<gazebo reference="base_link">
  <sensor name="bumper" type="contact">
    <always_on>true</always_on>
    <update_rate>50</update_rate>
    <topic>/bumper</topic>
    <contact><collision>base_link_collision</collision></contact>
  </sensor>
</gazebo>
```
In `launch/world.launch.py`, add to the bridge arguments:
```python
'/bumper@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts',
```
Check: drive into a wall with teleop, then `ros2 topic echo /bumper --once`.
If `ign topic -l | grep bumper` shows a different name (e.g. `/world/.../bumper/contact`), bridge that one and pass `--bumper_topic`.
If there is no `/bumper` publisher, the runner falls back to a geometric check
(base_link closer than 0.21 m to an obstacle on the map). That is conservative for tables,
because furniture is a solid bounding box on the map.

### b) Nothing else
The runner calibrates odom→world itself by teleporting the robot to the spawn pose,
so it works whether your OdometryPublisher odom starts at the spawn or at the world origin.

## 1. Validate with GT depth (smoke test)

```bash
cd ~/depth_safe_robot/mono_nav_worlds/eval
./run_batch.sh smoke          # test_02, gt, 3 episodes, RViz open
```
Or step by step in two terminals:
```bash
ros2 launch ./eval_stack.launch.py world:=test_02 method:=gt rviz:=true
python3 episode_runner.py --episodes episodes/test_02.yaml --method gt --out results_smoke --limit 3
```
What to check:
- the robot jumps to each start, the costmaps clear, then it drives to the goal;
- `results_smoke/results.csv` shows `success` for easy episodes;
- push the robot into a wall once (teleop while an episode runs): outcome must be `collision`.

## 2. Evaluation: GT and mono raw on test worlds

```bash
./run_batch.sh eval                       # 4 worlds × 2 methods × 30 episodes = 240 runs
RTF=0.5 ./run_batch.sh eval               # if mono_depth is much slower than the camera rate
EXTRA_LAUNCH="gt_sync:=true" ./run_batch.sh eval   # GT gets same frames/latency as mono (fairer)
```
It resumes: finished `(method, episode)` rows in `results/results.csv` are skipped, so you can stop and restart.
At the end it prints `results/summary.md` (tables ready for the paper).

Use the same `RTF` / `gt_sync` for every method, and report them.

## 3. Collect training data on train worlds

```bash
./run_batch.sh collect
```
Both GT-driven and mono-driven runs on `train_01..05`. Mono runs include failures, which give
close-range views of obstacles where σ matters most. 30% of train goals sit close to obstacles.
Expect roughly 5–15k frames. For more, regenerate train episodes with `--pairs 80`, or add worlds
(`generate_world.py --name train_06 --seed 6 ...`). Never use `test_*` for training.

Output:
```
dataset/<world>/<method>/rgb/000123.png      RGB
dataset/<world>/<method>/depth/000123.png    uint16 mm, 0 = invalid
dataset/<world>/<method>/index.jsonl         frame, episode, stamp, odom pose
dataset/<world>/<method>/camera_info.json
```
Join `index.jsonl` with `results_train/results.csv` on (method, episode) to know each frame's outcome.

## Outcome definitions (runner)
- `success`: Nav2 SUCCEEDED and within 0.30 m of the goal
- `collision`: bumper touched anything except the floor (ends the run)
- `timeout`: sim time > `t_max = 2.5·L_A*/0.3 + 20 s`
- `stuck`: < 0.10 m progress in 40 s
- `nav_failed`: Nav2 aborted (no path, recoveries exhausted)

Episode difficulty: `easy` = line of sight, no hard case; `hard` = passes glass / blank wall / dark side / narrow gap; `medium` = rest.
`test_clean` has no hard cases, so its episodes are easy/medium only.

## Notes
- Furniture uses the footprints from `generate_world.py`; `Door_01` is approximated (0.95 × 0.12 m).
  The map is only used for sampling starts/goals, A* length, clearance, and the fallback collision check.
- `--spawn_z` (default 0.06) must match the `-z` used when spawning the robot.
- Teleport uses `ign service /world/<world>/set_pose` (UserCommands plugin, already in your worlds).

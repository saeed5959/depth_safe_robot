# mono_nav_worlds

Indoor worlds for the uncertainty-aware monocular navigation paper.
Gazebo **Fortress** / ROS 2 **Humble**.

## Contents

| Path | What |
|---|---|
| `worlds/train_01..05.sdf` | Training worlds (corridor + hall), varied hard cases |
| `worlds/test_clean.sdf` | Test world with **no** hard cases (baseline should mostly work here) |
| `worlds/test_01..03.sdf` | Test worlds with all hard cases (glass, blank wall, dark area) |
| `worlds/small_house.sdf` | AWS small house ported to Fortress (extra realistic world) |
| `worlds/*_goals.yaml` | Start pose and 4 goals per world, in world **and** odom frame |
| `previews/*.png` | Top view of every generated world |
| `tools/generate_world.py` | The generator (one seed = one world) |
| `tools/build_all.sh` | Regenerates all worlds with the fixed seeds |
| `launch/world.launch.py` | Starts Gazebo + mono_bot + bridge for any world |

## Setup (once)

```bash
cd ~/depth_safe_robot
unzip mono_nav_worlds.zip && cd mono_nav_worlds
./setup.sh            # downloads the AWS furniture models from GitHub (~80 MB)
```

## Run

```bash
ros2 launch ./launch/world.launch.py world:=test_clean
ros2 launch ./launch/world.launch.py world:=test_01
ros2 launch ./launch/world.launch.py world:=small_house x:=0 y:=0
# another URDF location:
ros2 launch ./launch/world.launch.py world:=train_01 urdf:=/path/to/mono_bot.urdf
```

The launch file publishes the same topics as `sim.launch.py` (camera, depth, camera_info,
odom, tf, cmd_vel), so the depth node, `depth_image_proc` and Nav2 work unchanged.

## World layout

```
start ▶ corridor (x 0..8, 1.8 m wide)  →  hall (x 8..15, y -4..4)
```
- Corridor: textured walls, baseboards, closed doors, pictures, cardboard box stacks, shoe rack.
- Hall: AWS furniture (sofa, TV cabinet, wardrobe, fridge, tables, chairs...), loose boxes.
- Hard cases (flags `--hard glass,blank,dark`):
  - `glass`: free-standing glass partition (2 m tall) with thin posts
  - `blank`: a 2–3 m stretch of the corridor with plain white walls, plus a plain white panel in the hall
  - `dark`: one half of the hall lit at ~10 %

Goals per world: `corridor_end`, `hall_far`, `hall_left`, `hall_right`.
Every goal is guaranteed reachable for a robot of radius 0.30 m.

Nav2 runs in the `odom` frame, which starts at the spawn pose, so send goals from
`goals_odom` in the yaml (or convert `goals_world` by subtracting the start pose).

## Making more worlds

```bash
cd tools
python3 generate_world.py --name train_06 --seed 6 --hard glass,blank,dark
```
Rule for the paper: never train (or tune) on the `test_*` seeds.

## Things to check in Gazebo the first time

1. **Glass in the ground-truth depth camera.** Look at `/camera/depth_image` when the robot faces the
   glass. If Gazebo's depth sensor sees *through* it, the upper-bound baseline will also hit glass
   (similar to real depth sensors). Write down which case you get; it matters for the discussion.
2. **Lighting.** If the hall is too dark or too bright, change `point_light` intensities in
   `generate_world.py` and rerun `build_all.sh`.
3. Run `compare_depth` in `test_clean`: you want `median pred/gt` near 1.0 here.

## Credits

Furniture models and wall/floor base textures: AWS RoboMaker small house world (MIT-0),
via the Gazebo Harmonic port by the Intelligent Robotics Lab (URJC).

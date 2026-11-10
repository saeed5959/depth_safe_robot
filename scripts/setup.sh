#!/usr/bin/env bash
# One-time setup after cloning:
#   - clones Depth-Anything-V2 into third_party/ and downloads the metric (Hypersim, ViT-S) checkpoint
#   - downloads the AWS RoboMaker small-house models (via mono_nav_worlds/setup.sh) and fixes the
#     ShoeRack model, whose inertia Gazebo Fortress rejects ("invalid inertia")
#   - fills the absolute paths in the Nav2 configs
#   - builds the ROS 2 package
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# 1) Depth Anything V2 + metric checkpoint
mkdir -p third_party
if [ ! -d third_party/Depth-Anything-V2 ]; then
  git clone https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
fi
CK=third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth
if [ ! -f "$CK" ]; then
  mkdir -p "$(dirname "$CK")"
  wget -O "$CK" "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
fi

# 2) world assets
if [ -x mono_nav_worlds/setup.sh ]; then (cd mono_nav_worlds && ./setup.sh); fi
SHOE=$(find mono_nav_worlds -path "*ShoeRack_01*" -name model.sdf | head -1 || true)
if [ -n "$SHOE" ] && ! grep -q "fixed_inertia" "$SHOE"; then
  python3 - "$SHOE" <<'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'<inertial>.*?</inertial>',
           '<inertial><!-- fixed_inertia --><mass>5.0</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy>'
           '<izz>0.1</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>', s, flags=re.S)
open(p, 'w').write(s); print('fixed inertia:', p)
EOF
fi

# 3) absolute paths
"$ROOT/scripts/configure_paths.sh"

# 4) build the ROS 2 package
set +u; source /opt/ros/humble/setup.bash; set -u
(cd ros2_ws && colcon build --symlink-install --packages-select mono_nav)

echo
echo "Done. In every new terminal:"
echo "  source /opt/ros/humble/setup.bash && source $ROOT/ros2_ws/install/setup.bash"
echo "  export DA=$ROOT/third_party/Depth-Anything-V2"

#!/usr/bin/env python3
"""Port the AWS small house world (gz-sim Harmonic version) to Gazebo Fortress.

usage: python3 port_small_house.py <small_house.world> <out.sdf>
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
s = s.replace("<sdf version='1.11'>", "<sdf version='1.9'>")
s = s.replace('gz-sim-', 'ignition-gazebo-').replace('gz::sim::', 'ignition::gazebo::')
s = re.sub(r"\s*frame=''", '', s)
s = s.replace("<world name='default'>", "<world name='small_house'>")
# ambient light so rooms are not too dark when the robot looks at the floor
s = s.replace('<physics name="1ms"', '<scene><ambient>0.4 0.4 0.4 1</ambient><shadows>true</shadows>'
              '<grid>false</grid></scene>\n    <physics name="1ms"', 1)
open(dst, 'w').write(s)
print('wrote', dst)

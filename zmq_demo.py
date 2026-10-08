"""Mini demo: how a Python program controls CoppeliaSim over the ZMQ Remote API (the same calls main.py uses).

Prerequisites: CoppeliaSim is open with scenes/Demo_CV.ttt and the simulation is STOPPED.
Run:  python zmq_demo.py
"""
import math

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# 1. Connect. CoppeliaSim acts as a server on localhost:23000; this program is the client.
client = RemoteAPIClient()           # host='localhost', port=23000
sim = client.require('sim')          # 'sim' is the family of functions you know from Lua: sim.getObject, sim.step ...
print('Simulation state (0 = stopped):', sim.getSimulationState())

# 2. Find objects by their path in the scene hierarchy. You get back integer HANDLES.
body = sim.getObject('/body')
left = sim.getObject('/body/J1')
right = sim.getObject('/body/J2')
cam = sim.getObject('/body/visionSensor')
print('Handles:', body, left, right, cam)

# 3. Read things from the scene.
x, y, z = sim.getObjectPosition(body, sim.handle_world)
print(f'Robot position: x={x:.2f} y={y:.2f} z={z:.2f}')
print('Camera resolution:', sim.getVisionSensorResolution(cam))

# 4. Take control of time: with stepping ON the simulation only advances when we call sim.step().
sim.setStepping(True)
sim.startSimulation()
dt = sim.getSimulationTimeStep()
print(f'One sim.step() advances {dt:.2f} s of simulated time')

# 5. Command the wheels (joint target velocities in rad/s) and advance 2 s of simulated time.
for _ in range(int(2.0 / dt)):
    sim.setJointTargetVelocity(left, 2.0)
    sim.setJointTargetVelocity(right, 2.0)
    sim.step()                       # nothing moves between two sim.step() calls
x2, y2, _ = sim.getObjectPosition(body, sim.handle_world)
print(f'After 2 s of both wheels at 2 rad/s: moved {math.hypot(x2 - x, y2 - y):.2f} m '
      f'(wheel radius 0.05 m x 2 rad/s x 2 s = {0.05 * 2 * 2:.2f} m expected)')
# Joint angles wrap around at +-pi (4.0 rad is reported as 4.0 - 2*pi = -2.28 rad). main.py therefore uses the
# difference between two readings, wrapped again (function angle_delta), never the raw angle.
print(f'Wheel angle read back (what the odometry uses): left = {sim.getJointPosition(left):.2f} rad  (commanded 2 rad/s x 2 s = 4.00 rad)')

# 6. Read a camera picture and save it (CoppeliaSim sends rows bottom-up, RGB; OpenCV wants top-down, BGR).
sim.setJointTargetVelocity(left, 0)
sim.setJointTargetVelocity(right, 0)
sim.step()
raw, (w, h) = sim.getVisionSensorImg(cam)
rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
cv2.imwrite('zmq_demo.png', cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR))
print('Saved zmq_demo.png')

# 7. Always leave the simulator clean.
sim.stopSimulation()
sim.setStepping(False)

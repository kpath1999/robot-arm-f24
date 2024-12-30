import numpy as np
import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])

cube_start_pos = [0.5, 0, 0.05]
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orientation)

targid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
obj_of_focus = targid

numJoints = p.getNumJoints(targid)
print("numJoints=", numJoints)


for step in range(1000000):
    focus_pos, focus_orn = p.getBasePositionAndOrientation(obj_of_focus)
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focus_pos)
    p.stepSimulation()
    time.sleep(1. / 240.)


p.disconnect()

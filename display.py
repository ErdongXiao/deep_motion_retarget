import pybullet as p 
import time 
import pickle as pkl
from scipy.spatial.transform import Rotation as R
# class Dog:
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version 

planeId = p.loadURDF("plane.urdf") 

p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
urdfFlags = p.URDF_USE_SELF_COLLISION
# quadruped, index_map = p.loadURDF("aliengo/urdf/aliengo.urdf", [0,0,0.45], [0,0,0,1], flags = urdfFlags, useFixedBase=False), [5,6,7,1,2,3,9,10,11,13,14,15]
quadruped, index_map = p.loadURDF("go1/urdf/go1.urdf", [0,0,0.34], [0,0,0,1], flags = urdfFlags, useFixedBase=False), [7,9,10,2,4,5,17,19,20,12,14,15]
# file = open("mpc_ref_aliengo_walk_xvel=0.6.pkl", "rb")
# file = open("rl_ref_go1_bipedal_yaw.pkl", "rb")
# file = open("wtw_ref_go1_pacing.pkl", "rb")
# file = open("wtw_ref_go1_bounding.pkl", "rb")
# file = open("wtw_ref_go1_pronking.pkl", "rb")

motion_data = pkl.load(file)
# for i in range(p.getNumJoints(quadruped)):
#     print(p.getJointInfo(quadruped, i))


# print(motion_data.keys())
for i, joint_pos in enumerate(motion_data["joint_pos"]):
    for j in range(len(joint_pos)):
      p.resetJointState(quadruped, index_map[j], joint_pos[j])

    # body_quat = R.from_euler("xyz", motion_data["body_rpy"][i]).as_quat()
    # p.resetBasePositionAndOrientation(quadruped, motion_data["body_pos"][i], body_quat)
    print(motion_data["body_pos"][i])
    motion_data["body_pos"][i][2] += 1.
    p.resetBasePositionAndOrientation(quadruped, motion_data["body_pos"][i], motion_data["body_quat"][i])
    time.sleep(0.02)

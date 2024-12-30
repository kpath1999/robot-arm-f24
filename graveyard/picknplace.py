import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class PickAndPlaceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, GUI=False):
        super(PickAndPlaceEnv, self).__init__()
        
        # Define action and observation spaces
        # Example: [delta_x, delta_y, delta_z, gripper_action]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # Example observation: [ee_x, ee_y, ee_z, obj_x, obj_y, obj_z, gripper_state]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        # Initialize PyBullet
        if not GUI:
            self.physics_client = p.connect(p.DIRECT)  # Use GUI for visualization
        else:
            self.physics_client = p.connect(p.GUI)
        self.GUI = GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment components
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], p.getQuaternionFromEuler([0,0,0]))
        self.object_id = p.loadURDF("cube_small.urdf", [0.5, 0, 0.05])
        
        # Identify end-effector and gripper joints
        self.ee_link_index = 11  # Example for Franka Panda
        self.gripper_joint_indices = [9, 10]  # Example joint indices for gripper
        
    def reset(self, seed=None, **kwargs):

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], p.getQuaternionFromEuler([0,0,0]))
        self.object_id = p.loadURDF("cube_small.urdf", [0.5, 0, 0.05])
        
        # Reset robot joints
        num_joints = p.getNumJoints(self.robot_id)
        for joint in range(num_joints):
            p.resetJointState(self.robot_id, joint, 0)
        
        # Initial observation
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Get end-effector position
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = ee_state[4]
        
        # Get object position
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        # Get gripper state
        gripper_states = [p.getJointState(self.robot_id, idx)[0] for idx in self.gripper_joint_indices]
        gripper_closed = 1.0 if sum(gripper_states) > 0 else 0.0
        
        observation = np.array(list(ee_pos) + list(obj_pos) + [gripper_closed], dtype=np.float32)
        return observation
    
    def step(self, action):
        truncated = False
        # Scale and apply actions
        try:
            delta_pos = action[:3] * 0.05  # Scale movement
            gripper_action = (action[3] + 1) / 2  # Scale from [-1,1] to [0,1]
            
            # Get current end-effector position
            ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
            current_pos = np.array(ee_state[4])
            target_pos = current_pos + delta_pos
            
            # Inverse kinematics to get joint angles
            joint_angles = p.calculateInverseKinematics(self.robot_id, self.ee_link_index, target_pos)
            
            # Apply joint angles
            for i, angle in enumerate(joint_angles):
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=angle,
                                        force=500)
            
            # Control gripper
            for idx in self.gripper_joint_indices:
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=idx,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=gripper_action,
                                        force=50)
            
            # Step simulation
            p.stepSimulation()
        
        # Get new observation
            obs = self._get_obs()
            
            # Compute reward
            reward, done = self._compute_reward()
        except:
            truncated = True
            obs = self._get_obs()
            reward = -100
            done = False
            
        return obs, reward, done, truncated, {}
    
    def _compute_reward(self):
        # Example reward: negative distance to target
        target_pos = np.array([0.6, 0, 0.05])  # Define target placement random numbers right now, but we can change to the location of the table or whatever
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        distance = np.linalg.norm(np.array(obj_pos[:3]) - target_pos)
        
        # Reward shaping
        reward = -distance
        done = False
        success_threshold = 0.02  # Define success criteria
        
        if distance < success_threshold:
            reward += 100  # Reward for successful placement
            done = True
        
        return reward, done
    
    def render(self, mode='human'):
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        else:
            pass
    
    def close(self):
        p.disconnect()

from stable_baselines3.common.env_checker import check_env

env = PickAndPlaceEnv()
check_env(env)
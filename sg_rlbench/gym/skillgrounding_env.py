from typing import Union, Dict, Tuple

import gym
import numpy as np
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from sg_rlbench.action_modes.action_mode import MoveArmThenGripper
from sg_rlbench.action_modes.arm_action_modes import JointVelocity,JointPosition

from sg_rlbench.action_modes.gripper_action_modes import Discrete, Discrete3
from sg_rlbench.environment import Environment
from sg_rlbench.observation_config import ObservationConfig

class SkillGroundingEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, task_class, observation_mode='state',
                 render_mode: Union[None, str] = None, gripper_action_mode:str='discrete'):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(True)
            img_size = (224,224)
            obs_config.right_shoulder_camera.image_size = img_size
            obs_config.left_shoulder_camera.image_size = img_size
            obs_config.overhead_camera.image_size = img_size
            obs_config.wrist_camera.image_size = img_size
            obs_config.front_camera.image_size = img_size
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)
        
        obs_config.joint_velocities = False
        obs_config.joint_forces = False
        
        self.gripper_action_mode = gripper_action_mode
        # if self.gripper_action_mode == "continuous":
        #     action_mode = MoveArmThenGripper(JointPosition(absolute_mode=False), Continuous(gripper_action_scale=gripper_action_scale))
        # else:
        #     action_mode = MoveArmThenGripper(JointPosition(absolute_mode=False), Discrete3())
        # action_mode = MoveArmThenGripper(JointPosition(absolute_mode=False), Discrete())
        action_mode = MoveArmThenGripper(JointPosition(absolute_mode=False), Discrete3())
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)
        
        _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.env.action_shape)

        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
        elif observation_mode == 'vision':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                })
        self.waypoints = self.task._task.get_waypoints()
        self.total_timesteps = 0
        self.expert = True

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        elif self._observation_mode == 'vision':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "wrist_rgb": obs.wrist_rgb,
                "front_rgb": obs.front_rgb,
            }

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            frame = self._gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame

    def reset(self) -> Dict[str, np.ndarray]:
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        # obs, reward, terminate = self.task.step(action)
        # return self._extract_obs(obs), reward, terminate, {}
        action = action.copy()
        info, waypoint_error = dict(), False
        self.total_timesteps += 1
        joint_position = self.get_joint_positions()
        target_joint_position = self.get_joint_target_positions()

        if self.expert:
            delta_action = target_joint_position - joint_position
            # delta_action = delta_action / (np.max(delta_action) / 9 + 1e+6)
            delta_action = np.clip(delta_action, a_max=.1, a_min=-.1)
            # print(np.max(np.abs(delta_action)))
            action[:7] = delta_action.copy()
            self.gripper_action = action[7].copy()
            info['expert_action'] = np.concatenate([action[:7], self._transform_gripper_action], axis=0)
            action = info['expert_action']

        obs, reward, terminate = self.task.step(action)
        success, terminate = self.task.success()
        info["is_success"] = success
        return self._extract_obs(obs), reward, terminate, info
    def close(self) -> None:
        self.env.shutdown()

    @property
    def _transform_gripper_action(self) -> float:
        if self.gripper_action_mode == 'continuous':
            return np.array([self.gripper_action])
        else:
            if self.gripper_action == 0.1:
                return np.array([-0.1, -0.1, 0.1])
            elif self.gripper_action == -0.1:
                return np.array([0.1, -0.1, -0.1])
            else:
                return np.array([-0.1, 0.1, -0.1])

    @property
    def _robot_arm(self):
        return self.task._task.robot.arm
                

    def get_joint_positions(self):
        return np.array(self._robot_arm.get_joint_positions())

    def get_joint_target_positions(self):
        return np.array(self._robot_arm.get_joint_target_positions())


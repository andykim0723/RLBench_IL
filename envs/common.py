import gym
import d4rl
import numpy as np
from jaxbc.utils.jaxbc_utils import yielding

from sg_rlbench.action_modes.action_mode import MoveArmThenGripper
from sg_rlbench.action_modes.arm_action_modes import JointVelocity
from sg_rlbench.action_modes.gripper_action_modes import Discrete
from sg_rlbench.environment import Environment

import sg_rlbench.gym

def set_env(cfg):
    if cfg['env_name'] == "rlbench":
        obs_type = "vision"
        env = gym.make(cfg['task_name'] + '-' + obs_type + '-v0')

        return env

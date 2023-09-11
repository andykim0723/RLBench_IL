import gym
import numpy as np
from jaxbc.utils.jaxbc_utils import yielding

from sg_rlbench.action_modes.action_mode import MoveArmThenGripper
from sg_rlbench.action_modes.arm_action_modes import JointVelocity
from sg_rlbench.action_modes.gripper_action_modes import Discrete
from sg_rlbench.environment import Environment
from sg_rlbench.tasks.pick_and_lift_simple import PickAndLiftSimple
from sg_rlbench.gym.skillgrounding_env import SkillGroundingEnv

import sg_rlbench.gym

def set_env(cfg):
    if cfg['env_name'] == "rlbench":
        obs_type = "vision"
        # env = gym.make(cfg['task_name'] + '-' + obs_type + '-v0')
        task = PickAndLiftSimple
        env = SkillGroundingEnv(task_class=task, render_mode='rgb_array')
        env.expert = False
        return env

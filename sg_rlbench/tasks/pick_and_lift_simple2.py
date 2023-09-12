from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.backend._sim_cffi import lib
from pyrep import PyRep

from sg_rlbench.backend.task import Task
from sg_rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from sg_rlbench.backend.spawn_boundary import SpawnBoundary
from sg_rlbench.backend.robot import Robot
from sg_rlbench.const import colors

from sg_rlbench.action_modes.action_mode import MoveArmThenGripper
from sg_rlbench.action_modes.arm_action_modes import JointVelocity,JointPosition
from sg_rlbench.action_modes.gripper_action_modes import Discrete

class PickAndLiftSimple2(Task):

    # def __init__(self, pyrep: PyRep, robot: Robot, name: str = None):
    #     super().__init__(pyrep,robot,name)
    #     self.colors = colors[:5]

    def init_task(self) -> None:
        # arm_init_pos = [0.05,-0.15,-0.085,-2.2,0.15,2.05,-0.03]
        # self.robot.arm.set_joint_target_positions(arm_init_pos)

        # arm_init_pos = [0.007436040788888931, -0.14280149340629578, 0.009935889393091202, -2.498753070831299, 0.0019878200255334377, 2.2860946655273438, 0.8013144731521606]
        # self.robot.arm.set_joint_positions(arm_init_pos)

        self.target_pos = ["left","right"] 

		# objects
        self.target_block = Shape('pick_and_lift_target') 
        self.distractor = Shape('stack_blocks_distractor')
        self.register_graspable_objects([self.target_block])

		# spawn boundaries
        self.boundary_left = SpawnBoundary([Shape('pick_and_lift_boundary_left')])
        self.boundary_right = SpawnBoundary([Shape('pick_and_lift_boundary_right')])
        
		# success detector
        self.success_detector = ProximitySensor('pick_and_lift_success')
        self.success_detector.set_position([0.0,0.0,-0.15], relative_to=self.target_block,reset_dynamics=False)

        # self.success_detector.set_position([1.0,1.0,0.55], relative_to=self.target_block,reset_dynamics=False)

		# success conditions
        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.target_block),
            DetectedCondition(self.target_block, self.success_detector)
        ])
        self.register_success_conditions([cond_set])


    def init_episode(self, index: int) -> List[str]:
        # check pick up red block is runnable
        # change size
        # lib.simScaleObject(self.target_block._handle,0.5,0.5,0.5,0)

        # block_color_name, block_rgb = self.colors[index]

        # self.target_block.set_color(block_rgb)
        # color_choice = np.random.choice( 
        #     list(range(index)) + list(range(index + 1, len(self.colors))),
        #     size=1, replace=False)[0]
        # name, rgb = self.colors[color_choice]

        # self.distractor.set_color(rgb)

        # self.boundary_left.clear()
        # self.boundary_right.clear()

        # target_pos = np.random.choice(self.target_pos)
        # target_pos = "left"
        # if target_pos == "left":
        #     self.boundary_left.sample(
        #         self.target_block, min_rotation=(0.0, 0.0, 0.0),
        #         max_rotation=(0.0, 0.0, 0.0))           
        #     self.boundary_right.sample(
        #         self.distractor, min_rotation=(0.0, 0.0, 0.0),
        #         max_rotation=(0.0, 0.0, 0.0))
        # elif target_pos == "right":
        #     self.boundary_left.sample(
        #         self.distractor, min_rotation=(0.0, 0.0, 0.0),
        #         max_rotation=(0.0, 0.0, 0.0))           
        #     self.boundary_right.sample(
        #         self.target_block, min_rotation=(0.0, 0.0, 0.0),
        #         max_rotation=(0.0, 0.0, 0.0))

        # self.boundary_left.clear()
        # self.boundary_right.clear()        
        block_color_name = self.set_simple_task()

        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def variation_count(self) -> int:
        return 1
        # return len(self.colors)

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.concatenate([self.target_block.get_position(), self.success_detector.get_position()], 0)

    def is_static_workspace(self) -> bool:
            return True

    def set_simple_task(self):

        rotation = np.array([0.0,0.0,0.0])
        left_pos = np.array([0.225,-0.175,0.775])
        right_pos = np.array([0.225,0.175,0.775])
        target_block_color_name, red_rgb = colors[0] # red
        _, blue_rgb = colors[4] # blue

        self.target_block.set_color(red_rgb)
        self.target_block.set_position(right_pos)
        self.target_block.rotate(list(rotation))
        self.distractor.set_color(blue_rgb)
        self.distractor.set_position(left_pos)
        self.distractor.rotate(list(rotation))

        # arm_init_pos = [0.007163528352975845, -0.10803624987602234, 0.008292946964502335, -2.186707019805908, 0.0010930350981652737, 2.0084352493286133, 0.8003145456314087]
        # arm_init_pos = [0.007436040788888931, -0.14280149340629578, 0.009935889393091202, -2.498753070831299, 0.0019878200255334377, 2.2860946655273438, 0.8013144731521606]
        # self.robot.arm.set_joint_target_positions(arm_init_pos)
        # print(np.array(self.robot.arm.get_joint_positions())-np.array(arm_init_pos))
        # print(self.robot.arm.get_joint_positions())
        # print(self.robot.arm.get_joint_target_positions())
        # jp = [0.007437232881784439, -0.14239856600761414, 0.009938273578882217, -1.9623231887817383, 0.0019825748167932034, 2.2853682041168213, 0.8013125658035278] 
        # jtp = [0.007436040788888931, -0.14280149340629578, 0.009935889393091202, -2.498753070831299, 0.0019878200255334377, 2.2860946655273438, 0.8013144731521606] 

        jp = [0.007407668977975845, -0.14495226740837097, 0.00987008586525917, -2.496584415435791, 0.0019749454222619534, 2.282616376876831, 0.8012551069259644]
        jtp = [0.007436040788888931, -0.14280149340629578, 0.009935889393091202, -2.498753070831299, 0.0019878200255334377, 2.2860946655273438, 0.8013144731521606]
        # self.robot.arm.set_joint_target_positions(jtp)
        self.robot.arm.set_joint_positions(jp,disable_dynamics=True)




        return target_block_color_name

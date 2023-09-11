
import cv2
import numpy as np

def save_video(video_path,frames):
    print("video recording..")
    height, width, layers  = frames[0].shape
    size = (width,height)
    fps = 15
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), size)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()

def transform_states(states):
    
    idxes = {}
    idxes['gripper_open'] = (0,)
    idxes['joint_velocities'] = (1,8)
    idxes['joint_positions'] = (8,15)
    idxes['joint_forces'] = (15,22)
    idxes['gripper_pose'] = (22,29)
    idxes['gripper_joint_positions'] = (29,31)
    idxes['gripper_touch_forces'] = (31,37)
    idxes['task_low_dim_state'] = (37,43)

    go_idx = idxes['gripper_open']
    jv_idx = idxes['joint_velocities']
    jp_idx = idxes['joint_positions']
    jf_idx = idxes['joint_forces']
    gp_idx = idxes['gripper_pose']
    gjp_idx = idxes['gripper_joint_positions']
    gtf_idx = idxes['gripper_touch_forces']
    tld_idx = idxes['task_low_dim_state']
    

    if states.ndim == 2:
        gripper_open = states[:,go_idx[0]][:,np.newaxis]
        joint_velocities = states[:,jv_idx[0]:jv_idx[1]]
        joint_positions = states[:,jp_idx[0]:jp_idx[1]]
        joint_forces = states[:,jf_idx[0]:jf_idx[1]]
        gripper_pose = states[:,gp_idx[0]:gp_idx[1]]
        gripper_joint_positions = states[:,gjp_idx[0]:gjp_idx[1]]
        gripper_touch_forces = states[:,gtf_idx[0]:gtf_idx[1]]
        task_low_dim_state = states[:,tld_idx[0]:tld_idx[1]]

    elif states.ndim == 1:
        gripper_open = states[go_idx[0]]
        joint_velocities = states[jv_idx[0]:jv_idx[1]]
        joint_positions = states[jp_idx[0]:jp_idx[1]]
        joint_forces = states[jf_idx[0]:jf_idx[1]]
        gripper_pose = states[gp_idx[0]:gp_idx[1]]
        gripper_joint_positions = states[gjp_idx[0]:gjp_idx[1]]
        gripper_touch_forces = states[gtf_idx[0]:gtf_idx[1]]
        task_low_dim_state = states[tld_idx[0]:tld_idx[1]]

    # states = np.hstack([joint_positions,gripper_open])    
    states = np.hstack([gripper_open,joint_positions,gripper_pose,gripper_joint_positions,
                        gripper_touch_forces])    

    return states
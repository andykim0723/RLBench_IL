
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

    if states.ndim == 2:
        joint_positions = states[:,8:15]
        gripper_open = states[:,0][:,np.newaxis]
        # task_low_dim_state = sensors[:,-6]
        states = np.hstack([joint_positions,gripper_open[:np.newaxis]])
    elif states.ndim == 1:
        joint_positions = states[8:15]
        gripper_open = states[0]
        # task_low_dim_state = sensors[-6]
        states = np.hstack([joint_positions,gripper_open])       
    return states
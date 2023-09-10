import sg_rlbench.gym
from PIL import Image
import cv2
import numpy as np
import jax.numpy as jnp

from jaxbc.utils.common import transform_states

def rlbench_evaluate(env,fe_model_info,policy,num_episodes):

    episode_length = 220
    num_success = 0
    frames = {f"episode{k}":[] for k in range(num_episodes)}
    for i in range(num_episodes):
        obs = env.reset()

        # print(env.task.robot.arm.get_joint_positions())
        # state = obs['state'][-6:] # last 6 dim represent task_low_dim_state 
        state = obs['state']
        state = transform_states(state)

        img = obs['front_rgb']
        np_img = img.astype(np.float64)
        np_img /= 255.0
        np_img = np.expand_dims(np_img,axis=0)
        if fe_model_info:
            fe_model,variables = fe_model_info 
            img_embedding = fe_model.apply(variables,
                np_img,
                mutable=False)
            observations = np.concatenate(state,jnp.squeeze(img_embedding),axis=0)
        else:
            observations = state

        for j in range(episode_length):
            action = policy.predict(observations)
            obs, reward, terminate, _ = env.step(action)

            state = obs['state']
            state = transform_states(state)

            img = obs['front_rgb']
            np_img = img.astype(np.float64)
            np_img /= 255.0
            np_img = np.expand_dims(np_img,axis=0)  

            if fe_model_info:
                img_embedding = fe_model.apply(variables,
                    np_img,
                    mutable=False)
                observations = np.concatenate(state,jnp.squeeze(img_embedding),axis=0)
            else:
                observations = state

            frames[f'episode{i}'].append(img)
            
            if terminate:
                num_success += 1
                break
        print(f"episode{i}: success: {terminate} ")
    succecss_rate = num_success/num_episodes
    
    return succecss_rate,frames


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
        observations = obs


        for j in range(episode_length):
            action = policy.predict(observations)
            obs, reward, terminate, _ = env.step(action)
            observations = obs

            img = env.render(mode='rgb_array')
            frames[f'episode{i}'].append(img)
            
            if terminate:
                num_success += 1
                break
        print(f"episode{i}: success: {terminate} ")
    succecss_rate = num_success/num_episodes
    
    return succecss_rate,frames


import os
import json
import argparse
import re
import gym
import d4rl

import numpy as np
import pickle as pkl

from jaxbc.modules.trainer import BCTrainer,OnlineBCTrainer
from jaxbc.buffers.d4rlbuffer import d4rlBuffer
from jaxbc.buffers.rlbenchbuffer import RlbenchStateBuffer
from jaxbc.utils.jaxbc_utils import yielding,natural_keys
from jaxbc.utils.common import transform_states

from envs.common import set_env

def main(args):
 
    ### train info ###
    env_name, task_name = args.task.split('-')
    policy_name = args.policy
    print(f"train info -> task: {task_name} | policy: {policy_name} ")
    
    ### config file ###
    json_fname = task_name + '_' + policy_name
    config_filepath = os.path.join('configs',env_name,'train',json_fname+'.json')
    with open(config_filepath) as f:
        cfg = json.load(f)
    ### env ###
    env = set_env(cfg)

    if cfg['env_name'] == "rlbench":

        print("loading data..")
        data_path = cfg['info']['data_path'] 
        # load task_name 
        episodes_file_name = [path for path in os.listdir(data_path)]
        episodes_file_name.sort(key=natural_keys)

        episodes = []
        for file_name in episodes_file_name:
            episode = {}
            with open(data_path+"/"+file_name, 'rb') as f:
                episode_data = pkl.load(f)
            # images = episode_data['observations']['image']
            sensors = episode_data['observations']['sensor']
            actions = episode_data['actions'] 

            states = transform_states(sensors)

            episode['observations'] = states
            episode['actions'] = actions

        episodes.append(episode)                 
        
        if cfg['info']['use_img_embedding']:
            cfg['policy_args']['observation_dim'] += 512
        replay_buffer = RlbenchStateBuffer(cfg,env=env)
        replay_buffer.add_episodes_from_rlbench(episodes)

    trainer = BCTrainer(cfg=cfg)

    # train
    trainer.run(replay_buffer,env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Main Args
    
    parser.add_argument(
        "--task",type=str, default="d4rl-halfcheetah",
        choices=['d4rl-halfcheetah','d4rl-hopper','rlbench-pick_and_lift_simple'],
        required=True)

    parser.add_argument(
        "--policy",type=str, default="bc",
        choices=['bc'],
        required=True)
    
    args = parser.parse_args()
    main(args)



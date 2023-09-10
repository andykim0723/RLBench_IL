import os
import gym
import d4rl # need this for gym env creation
import json
import argparse
import numpy as np

from jaxbc.modules.low_policy.low_policy import MLPpolicy
from jaxbc.utils.common import save_video
from envs.common import set_env
from envs.eval_func import d4rl_evaluate, rlbench_evaluate

from jax_resnet import pretrained_resnet
import flax.linen as nn

def main(args):
    ### cfg ###
    
    config_filepath = os.path.join('configs','eval',args.mode+'.json')
    with open(config_filepath) as f:
        cfg = json.load(f)

    ### env ###
    # env = gym.make(cfg['env'])
    env = set_env(cfg)

    ### policy ###
    low_policy = MLPpolicy(cfg=cfg)
    load_path = os.path.join('logs',args.load_path)
    low_policy.load(load_path)

    ### evaluation ###
    if cfg['env_name'] == "d4rl":
        cfg['observation_dim'] = env.observation_space.shape
        cfg['action_dim'] = int(np.prod(env.action_space.shape))

        num_episodes = cfg['info']['eval_episodes']
        reward_mean = np.mean(d4rl_evaluate(env,low_policy,num_episodes))
        print("rewards: ", reward_mean)
    elif cfg['env_name'] == "rlbench":
        resnet18, variables = pretrained_resnet(18)
        model = resnet18()
        model = nn.Sequential(model.layers[0:-1])
        fe_model_info = (model,variables)
        num_episodes = cfg['info']['eval_episodes']
        success_rate,frames = rlbench_evaluate(env,fe_model_info,low_policy,num_episodes)
        video_path = "./"
        if video_path:       
            for k,v in frames.items():
                file_name = "evaluation" + "_" + k + ".mp4" 
                video_path = os.path.join(video_path,file_name)
                print("saving: ",file_name)
                save_video(video_path,v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",type=str, default="halfcheetah_bc",
        choices=['halfcheetah_bc','hopper_bc','pick_and_lift_simple_bc'])
    
    parser.add_argument(
        "--load_path",type=str, default="weights/hopper_bc/best")
    
    args = parser.parse_args()
    main(args)




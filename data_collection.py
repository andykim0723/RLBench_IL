import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
import tqdm
import cv2
import h5py
import numpy as np
import pickle as pkl
from PIL import Image
from multiprocessing import Process
from pyrep.errors import ConfigurationPathError

from sg_rlbench.gym.skillgrounding_env import SkillGroundingEnv
from sg_rlbench.tasks.pick_and_lift_simple import PickAndLiftSimple 


from itertools import product

DATA_PATH = '/home/andykim0723/SkillGrounding/data/'

def tbar_description(tbar, name, ext, waypoint_num, waypoint_len, timesteps, total_timesteps):
    tbar.set_description(name + '  ext: ' + ext + '  Waypoints %d/%d, Timesteps %d, Total_Timesteps %d'
                                             % (waypoint_num, waypoint_len, timesteps, total_timesteps))

def env_step(env, last_gripper, timesteps, total_timesteps, observation_sensors, observation_images, actions, rewards, terminals, infos):
    timesteps += 1
    total_timesteps += 1
    obs, reward, done, info = env.step(np.concatenate([np.zeros(7), np.array([last_gripper])], axis=0))

    img = env.render(mode='rgb_array')
    # Image.fromarray(img).save('test.png')
    observation_sensors.append(obs)
    observation_images.append(img)
    actions.append(info['expert_action'].copy())
    rewards.append(reward)
    terminals.append(False)
    del info['expert_action']
    infos.append(info)
    return timesteps, total_timesteps

def collect_episodic_data(task, num_episode,process_idx):
    seed = 12345 + process_idx
    np.random.seed(seed)
    tbar = tqdm.tqdm(range(num_episode))

    # task & gymenv creation
    task = PickAndLiftSimple
    env = SkillGroundingEnv(task_class=task, render_mode='rgb_array')
 
    for k in tbar:
        observations_sensors = []
        observations_images = []
        actions = []
        rewards = []
        terminals = []
        infos = []
        total_timesteps = 0

        while True:
            env.reset()
            timesteps = 0
            waypoints = env.waypoints
            success = False
            last_gripper = 0.1
            for i in range(len(env.waypoints)):
                pos = env.waypoints[i]._waypoint.get_position()
            # pos += np.random.normal(loc=0,scale=0.01,size=pos.shape)
            error_region = 0.02
            pos += np.clip(np.random.normal(size=pos.shape) * error_region, a_min=-error_region, a_max=error_region)
            env.waypoints[i]._waypoint.set_position(pos)

            while True:
                for p, point in enumerate(waypoints):
                    if point.skip:
                        continue
                    try:
                        path = point.get_path()
                    except ConfigurationPathError as e:
                        print(f'{env.task.get_name()} Could not get a path for waypoint {p}')
                        break
                    ext = point.get_ext()
                    done = False
                    while not done:
                        # add noise to waypoints 
                        # 0.03
                        # path._path_points += np.random.normal(loc=0,scale=0.00055,size=(path._path_points.size,))
                        done = path.step()
                        timesteps, total_timesteps = env_step(env, last_gripper, timesteps, total_timesteps, observations_sensors, observations_images, actions, rewards, terminals, infos)
                        tbar_description(tbar, env.task.get_name(), ext, p, len(waypoints)-1, timesteps, total_timesteps)
                        if done:
                            if len(ext) > 0:
                                contains_param = False
                                start_of_bracket = -1
                                gripper = env.task._task.robot.gripper

                                if 'open_gripper(' in ext:
                                    gripper.release()
                                    start_of_bracket = ext.index('open_gripper(') + 13
                                    contains_param = ext[start_of_bracket] != ')'\

                                    if not contains_param:
                                        last_gripper = 0.1
                                        timesteps, total_timesteps = env_step(env, last_gripper, timesteps, total_timesteps, observations_sensors, observations_images, actions, rewards, terminals, infos)
                                        tbar_description(tbar, env.task.get_name(), ext, p, len(waypoints)-1, timesteps, total_timesteps)
                                elif 'close_gripper(' in ext:
                                    start_of_bracket = ext.index('close_gripper(') + 14
                                    contains_param = ext[start_of_bracket] != ')'
                                    if not contains_param:
                                        last_gripper = -0.1
                                        timesteps, total_timesteps = env_step(env, last_gripper, timesteps, total_timesteps, observations_sensors, observations_images, actions, rewards, terminals, infos)
                                        tbar_description(tbar, env.task.get_name(), ext, p, len(waypoints) - 1, timesteps, total_timesteps)
                                if contains_param:
                                    rest = ext[start_of_bracket:]
                                    num = float(rest[:rest.index(')')])
                                    last_gripper = (num / 5) - 0.1
                                    timesteps, total_timesteps = env_step(env, last_gripper, timesteps, total_timesteps, observations_sensors, observations_images, actions, rewards, terminals, infos)
                                    tbar_description(tbar, env.task.get_name(), ext, p, len(waypoints) - 1, timesteps, total_timesteps)
                                if 'close_gripper(' in ext:
                                    for g_obj in env.task._task.get_graspable_objects():
                                        gripper.grasp(g_obj)
                success, _ = env.task._task.success()
                print("success: ",success)
                if not env.task._task.should_repeat_waypoints() or success:
                    break
            if not success:
                print(f'{env.task.get_name()} not success')
                total_timesteps -= timesteps
            else:
                print(f'{env.task.get_name()} success')
                data_dict = {}
                data_dict['observations'] = {}
                data_dict['observations']['sensor'] = np.array(observations_sensors)
                data_dict['observations']['image'] = np.array(observations_images)
                data_dict['actions'] = np.array(actions)
                data_dict['rewards'] = np.array(rewards)
                data_dict['terminals'] = np.array(terminals)
                data_dict['infos'] = infos

                if not os.path.exists(os.path.join(DATA_PATH, env.task.get_name())):
                    os.mkdir(os.path.join(DATA_PATH, env.task.get_name()))
                with open(os.path.join(DATA_PATH, env.task.get_name(), "episode_{}_{}.pkl".format(process_idx,k)), 'wb') as f:
                    pkl.dump(data_dict, f)
                break

if __name__ == '__main__':
    task_name = "pick_and_lift_simple-vision-v0"
    # collect_episodic_data(task_name, num_episode=3,process_idx=1)
    # exit()
    # for j in range(5):
    #     processes = [Process(target=collect_episodic_data, args=(task_name,10,j)) for i in range(5)]
    #     [t.start() for t in processes]
    #     [t.join() for t in processes]
    
    num_process = 5
    processes = [] 
    for i in range(num_process):
        p = Process(target=collect_episodic_data, args=(task_name,100,i+1))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

        # import RLBench.rlbench.gym
        # task_name = "pick_and_lift_simple-vision-v0"
        # collect_episodic_data(task_name, num_episode=5)
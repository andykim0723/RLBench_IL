import pickle as pkl
from sg_rlbench.gym.skillgrounding_env import SkillGroundingEnv
from sg_rlbench.tasks.pick_and_lift_simple import PickAndLiftSimple 
from sg_rlbench.backend.waypoints import Waypoint,Point
from jaxbc.utils.common import save_video
from PIL import Image
import numpy as np
from pyrep.objects.dummy import Dummy
import os 
def data_verify():


    # path = "/home/andykim0723/SkillGrounding/data/pick_and_lift_simple/episode_1_2.pkl"
    # with open(path,'rb') as f:
    #     data = pkl.load(f)
    #     sensor1 = data['observations']['sensor']

    # path = "/home/andykim0723/SkillGrounding/data/pick_and_lift_simple/episode_4_2.pkl"
    # with open(path,'rb') as f:
    #     data = pkl.load(f)
    #     sensor4 = data['observations']['sensor']
        
    # print(sensor4-sensor1)
    datapath = "/home/andykim0723/SkillGrounding/data/pick_and_lift_simple2"
    lengths = []
    for path in os.listdir(datapath):
        with open(os.path.join(datapath,path),'rb') as f:
            data = pkl.load(f)
            sensor = data['observations']['sensor']
            action = data['actions']
            lengths.append(sensor.shape[0])
            print('{}, {}'.format(sensor.shape[0]==action.shape[0],action.shape))

    print(max(lengths),min(lengths))
    exit()
    filename = 'episode_3_30'

    actions_list = []
    for i in range(50):
        path = "/home/andykim0723/SkillGrounding/data/pick_and_lift_simple/episode_1_{}.pkl".format(str(i+1))
        with open(path,'rb') as f:
            data = pkl.load(f)
        actions = data['actions']
        actions_list.append(actions)
    print(len(actions_list))
    actions_list = np.concatenate(actions_list)
    print(np.max(actions_list),np.min(actions_list))
    exit()

    for sensor in data['observations']['sensor']:
        print(sensor[-6:]) 
    print(data['observations']['sensor'].shape)
    exit()
    video_path = '{}.mp4'.format(filename)
    save_video(video_path=video_path,frames=data['observations']['image'])
    exit()
    print(data['observations']['image'].shape)
    print(data['actions'].shape)
    exit()
    task = PickAndLiftSimple
    env = SkillGroundingEnv(task_class=task, render_mode='rgb_array')
    env.expert = False

    env.reset()
    for action in data['actions']:
        obs, reward, terminate, _ = env.step(action)
        img = env.render(mode='rgb_array')
        Image.fromarray(img).save('test.png')

def check_state():
    path = "/home/andykim0723/jax_bc/data/pick_and_lift_simple/variation0/episodes/episode0/low_dim_obs.pkl"
    with open(path,'rb') as f:
        data = pkl.load(f)._observations[0]
    
    low_dim_data = data.get_low_dim_data()

    idxes = {}
    idxes['gripper_open'] = (0,)
    idxes['joint_velocities'] = (1,8)
    idxes['joint_positions'] = (8,15)
    idxes['joint_forces'] = (15,22)
    idxes['gripper_pose'] = (22,29)
    idxes['gripper_joint_positions'] = (29,31)
    idxes['gripper_touch_forces'] = (31,37)
    idxes['task_low_dim_state'] = (37,43)

    print(data.gripper_joint_positions)
    idx = idxes['gripper_joint_positions']
    print(low_dim_data[idx[0]:idx[1]])
    exit()

    gripper_open_idx = 0
    joint_vel_idx = (1,8)
    joint_pos_idx = (8,15)
    joint_force_idx = (...)
    grip_pos_idx = (...)
    grip_joint_pos_idx = (...)
    grip_touch_force = (...)
    task_low_dim = (...)

def waypoint_check():
    task = PickAndLiftSimple
    env = SkillGroundingEnv(task_class=task, render_mode='rgb_array')

    dummy = Dummy('waypoint0')

    dummy.set_position([0.2,0.0,1.0])

    point = Point(dummy, env.task._task.robot,
                start_of_path_func=None,
                end_of_path_func=None)
    
    path = point.get_path()


    done = False
    last_gripper = 0.1
    while not done:
        print(env.task._task.robot.arm.get_joint_positions())
        done = path.step()
        obs, reward, done, info = env.step(np.concatenate([np.zeros(7), np.array([last_gripper])], axis=0))
        img = env.render(mode='rgb_array')
        Image.fromarray(img).save('test.png')
    
    jp = env.task._task.robot.arm.get_joint_positions()
    jtp = env.task._task.robot.arm.get_joint_target_positions()
    print(jp)
    print(jtp)
    exit()
    while not done():
        
        img = env.render(mode='rgb_array')
        Image.fromarray(img).save('test.png')

    # create waypoint
    pass
if __name__ == '__main__':
    # waypoint_check()
    data_verify()
    # check_state()
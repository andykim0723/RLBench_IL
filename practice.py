import pickle as pkl


if __name__ == '__main__':
    path = "/home/andykim0723/SkillGrounding/data/pick_and_lift_simple/episode_0.pkl"
    with open(path,'rb') as f:
        data = pkl.load(f)
        print(data['observations'].keys())
        print(data['observations']['sensor'].shape)
        print(data['observations']['image'].shape)

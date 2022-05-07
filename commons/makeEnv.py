import gym 

def make_environment(env_type):
    if env_type == 'source':
        env = gym.make('CustomHopper-source-v0')
        return env
    elif env_type == 'target':
        env = gym.make('CustomHopper-target-v0')
        return env
    else:
        raise Exception("Invalid domain type selected. Please pick one among source - target")
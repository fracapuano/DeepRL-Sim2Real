import numpy as np
import pandas as pd
from pkg_resources import add_activation_listener
from tqdm import tqdm

import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from commons import trainModel, testModel, saveModel, makeEnv, utils
from commons.variant_callback import CustomCallback as CB
from env import *
from sb3_contrib.trpo.trpo import TRPO


def env_data_generation(timesteps, env, callback_): 

    agent = TRPO("MlpPolicy", env, verbose = 1)
    agent.learn(total_timesteps = timesteps, callback=callback_)

def dynamics_scoring(): 
    
    actionsDF = pd.read_csv("training_actions.txt", index_col = 0)
    rewardsDF = pd.read_csv("training_rewards.txt", index_col = 0)

    episodes, _ = actionsDF.shape
    frac = 0.5
    window = int(frac * episodes)

    actions_ = actionsDF.loc[window:, "action_measure"]
    rewards_ = rewardsDF.loc[window:, "episode_reward"]
    number_of_steps_ = rewardsDF.loc[window:, "number_of_steps"]

    action_weight, reward_weight, n_steps_weight = 0.4, 0.4, 0.2

    os.remove("training_actions.txt")
    os.remove("training_rewards.txt")

    with open("training_actions.txt", "w") as action_file: 
        action_file.write("episodeID,action_measure\n")
    with open("training_rewards.txt", "w") as reward_file: 
        reward_file.write("episodeID,episode_reward,number_of_steps\n")
    
    return action_weight * actions_.mean() + reward_weight * rewards_.mean() + n_steps_weight * number_of_steps_.mean()
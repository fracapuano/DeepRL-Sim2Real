import numpy as np
from tqdm import tqdm

import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from commons import trainModel, testModel, saveModel, makeEnv, utils
from commons.custom_callback import CustomCallback as CB
from callback_utils import env_data_generation, dynamics_scoring
from env import *
from sb3_contrib.trpo.trpo import TRPO

low, high = 0.25, 5
parametrization = [low, high, low, high, low, high]

env = makeEnv.make_environment("source")

n_samples = 100
n_params = 3
callback_ = CB()
timesteps = 250000

observations = np.zeros((n_samples, n_params + 1))
env.set_parametrization(parametrization)

with open("training_actions.txt", "w") as action_file: 
    action_file.write("episodeID,action_measure\n")
with open("training_rewards.txt", "w") as reward_file: 
    reward_file.write("episodeID,episode_reward,number_of_steps\n")

for s in tqdm(range(n_samples)):     
    env.set_random_parameters()

    agent = TRPO("MlpPolicy", env, verbose = 1)
    masses = env.sim.model.body_mass[2:]

    env_data_generation(timesteps = timesteps, env = env, callback_ = callback_)

    dynamics_score = dynamics_scoring()
    row = np.append(masses, dynamics_score)

    observations[s, :] = row

    del agent
    del row
    del masses

os.remove("training_actions.txt")
os.remove("training_rewards.txt")

np.savetxt("dynamics.txt", observations)
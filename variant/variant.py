import numpy as np
from tqdm import tqdm

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from commons import trainModel, testModel, saveModel, makeEnv, utils
from env import *
from sb3_contrib.trpo.trpo import TRPO

SEED = 42

low, high = 0.25, 15
parametrization = [low, high, low, high, low, high]

env = makeEnv.make_environment("source")

n_samples = 20
n_params = 3

observations = np.zeros((n_samples, n_params + 1))
print(observations.shape)

for s in tqdm(range(n_samples)): 
    agent = TRPO("MlpPolicy", env, verbose = 1, seed = SEED)

    env.set_parametrization(parametrization)
    env.set_random_parameters()
    masses = env.sim.model.body_mass[2:]
    print(masses)

    agent.learn(total_timesteps = 5)

    saveModel.save_model(agent, "trpo", folder_path="./variant")
    total_reward = testModel.test(
        agent, agent_type = "trpo", env=env, episodes = 50, render_bool = False, model_info = "./variant/trpo-model.mdl")
    
    row = np.append(masses, total_reward)

    observations[s, :] = row

    del agent
    os.remove("variant/trpo-model.mdl")
    del row
    del masses

np.savetxt("variant/dynamics.txt", observations)
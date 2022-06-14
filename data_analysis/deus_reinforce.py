import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tqdm import tqdm
import torch
import argparse
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from commons import makeEnv, saveModel, testModel, trainModel, utils
from agents import agentReinforce
from policies import tibNET

SEEDS = [42, 777, 299266, 303489, 295366]

ALGS = [
    "reinforce_baseline",
    "reinforce_standard",
    "reinforce_togo"
] 

MAP = dict.fromkeys(ALGS)

EPISODES = [50000 for _ in range(3)]

# ENV SETUP
source_env = makeEnv.make_environment("source")
target_env = makeEnv.make_environment("target")

observation_space_dim = source_env.observation_space.shape[-1]
action_space_dim = source_env.action_space.shape[-1]

# AGENTS INITIALIZATION
#DEFAULT REINFORCES
policy = tibNET.ReinforcePolicy(
    state_space=observation_space_dim,
    action_space=action_space_dim,
    )

reinforce_baseline = agentReinforce.Agent(
    policy,
    return_flag='baseline'
)

MAP["reinforce_baseline"] = reinforce_baseline

reinforce_standard = agentReinforce.Agent(
    policy,
    return_flag='standard'
)

MAP["reinforce_standard"] = reinforce_standard

reinforce_togo = agentReinforce.Agent(
    policy,
    return_flag='reward_to_go'
)

MAP["reinforce_togo"] = reinforce_togo

#PERFORMING TRAINING FOR 5 DIFFERENT RANDOM SEEDS
for idx in range(5):
    current_seed = source_env.seed(seed=SEEDS[idx])
    for agent, agent_name, duration in zip(MAP, ALGS, EPISODES):
        print(f"Training {agent_name} for {duration} episodes/timesteps. SEED = {current_seed}")

        trainModel.train(
            agent=MAP[agent_name],
            agent_type="reinforce",
            env=source_env,
            actorCriticCheck=False,
            batch_size=0,
            episodes=duration,
            print_every=100,
            print_bool=False,
            file_name=f"{current_seed}_{agent_name}",
            save_to_file_bool=True,
            info_file_path=f"./{agent_name}/",
        )

        saveModel.save_model(
            agent=MAP[agent_name],
            agent_type="reinforce",
            folder_path=f"./{agent_name}/"
        )
# SEEDLESS RETURNS
for alg in ALGS:

    with open(f"reinforce_sumup/seedless_{alg}_rewards.txt", "w") as seedless_file_header_reward:
        seedless_file_header_reward.write("Episode,Return\n")

    for seed in SEEDS:
        with open(f"reinforce_sumup/seedless_{alg}_rewards.txt", "a+") as seedless_file_reward:
            f = open(f"{alg}/{[seed]}_{alg}_reward_file.txt", "r")
            next(f) # salto l'header, non mi interessa averne più di uno.
            seedless_file_reward.write(f.read())
            seedless_file_reward.seek(0)
            f.seek(0)

#SEEDLESS ACTIONS
for alg in ALGS:

    with open(f"reinforce_sumup/seedless_{alg}_actions.txt", "w") as seedless_file_header_action:
        seedless_file_header_action.write("Episode,ActionMeasure\n")

    for seed in SEEDS:
        with open(f"reinforce_sumup/seedless_{alg}_actions.txt", "a+") as seedless_file_action:
            f = open(f"{alg}/{[seed]}_{alg}_action_file.txt", "r")
            next(f) # salto l'header, non mi interessa averne più di uno.
            seedless_file_action.write(f.read())
            seedless_file_action.seek(0)
            f.seek(0)

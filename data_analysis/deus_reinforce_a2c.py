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
from agents import agentReinforce, agentActorCritic
from policies import tibNET

ALGS = [
    "actorcritic",
    "reinforce"
]

SEEDS = [42, 777, 299266, 303489, 295366]

EPISODES = [20000 for _ in range(2)]

with open("../step2_tests/best_config.txt", "r") as a2c_best_config_file:
    a2c_best_config = json.load(a2c_best_config_file)

with open("../step2_tests/best_config.txt", "r") as reinforce_best_config_file:
    reinforce_best_config = json.load(reinforce_best_config_file)

batch_size_a2c=a2c_best_config["configurations"]["actorCritic"]["batch_size"]

source_env = makeEnv.make_environment("source")

observation_space_dim = source_env.observation_space.shape[-1]
action_space_dim = source_env.action_space.shape[-1]

#PERFORMING TRAINING FOR 5 DIFFERENT RANDOM SEEDS FOR BEST_CONFIG_A2C
for idx in range(5):
    MAP = dict.fromkeys(ALGS)
    current_seed = source_env.seed(seed=SEEDS[idx])

        #A2C BEST CONFIG
    policy_a2c = tibNET.ActorCriticPolicy(
        state_space=observation_space_dim,
        action_space=action_space_dim,
        hidden=a2c_best_config["configurations"]["actorCritic"]["n_neurons"],
        init_sigma=a2c_best_config["configurations"]["actorCritic"]["sigma"]
        )

    agent_a2c = agentActorCritic.Agent(
        policy=policy_a2c,
        net_type="tibNET",
        gamma=a2c_best_config["configurations"]["actorCritic"]["gamma"],
        lr=a2c_best_config["configurations"]["actorCritic"]["lr"]
        )

    MAP["actorcritic"] = agent_a2c

    policy_reinforce = tibNET.ReinforcePolicy(
        state_space=observation_space_dim,
        action_space=action_space_dim,
        hidden=reinforce_best_config["configurations"]["reinforce"]["n_neurons"],
        init_sigma=reinforce_best_config["configurations"]["reinforce"]["sigma"]
    )
    agent_reinforce = agentReinforce.Agent(
        policy=policy_reinforce,
        gamma=reinforce_best_config["configurations"]["reinforce"]["gamma"],
        lr=reinforce_best_config["configurations"]["reinforce"]["lr"],
        return_flag="baseline"
    )

    MAP["reinforce"] = agent_reinforce

    for agent, agent_name, duration in zip(MAP, ALGS, EPISODES):
        print(f"Training {agent_name} for {duration} episodes. SEED = {current_seed}")

        if agent_name.lower() == "actorCritic":
            actorCriticCheck=True
        else:
            actorCriticCheck=False

        trainModel.train(
            agent=MAP[agent_name],
            agent_type=agent_name,
            env=source_env,
            actorCriticCheck=actorCriticCheck,
            batch_size=batch_size_a2c,
            episodes=duration,
            print_every=duration,
            print_bool=False,
            file_name=f"{current_seed}_{agent_name}",
            save_to_file_bool=True,
            info_file_path=f"./a2c_reinforce/",
        )

        saveModel.save_model(
            agent=MAP[agent_name],
            agent_type=agent_name,
            folder_path=f"./a2c_reinforce/"
        )
    del agent_a2c
    del agent_reinforce
    del policy_a2c
    del policy_reinforce
    del MAP

# SEEDLESS RETURNS
for alg in ALGS:

    with open(f"a2c_reinforce_sumup/seedless_{alg}_rewards.txt", "w") as seedless_file_header_reward:
        seedless_file_header_reward.write("Episode,Return\n")

    for seed in SEEDS:
        with open(f"a2c_reinforce_sumup/seedless_{alg}_rewards.txt", "a+") as seedless_file_reward:
            f = open(f"a2c_reinforce/{[seed]}_{alg}_reward_file.txt", "r")
            next(f) # salto l'header, non mi interessa averne più di uno.
            seedless_file_reward.write(f.read())
            seedless_file_reward.seek(0)
            f.seek(0)

#SEEDLESS ACTIONS
for alg in ALGS:

    with open(f"a2c_reinforce_sumup/seedless_{alg}_actions.txt", "w") as seedless_file_header_action:
        seedless_file_header_action.write("Episode,ActionMeasure\n")

    for seed in SEEDS:
        with open(f"a2c_reinforce_sumup/seedless_{alg}_actions.txt", "a+") as seedless_file_action:
            f = open(f"a2c_reinforce/{[seed]}_{alg}_action_file.txt", "r")
            next(f) # salto l'header, non mi interessa averne più di uno.
            seedless_file_action.write(f.read())
            seedless_file_action.seek(0)
            f.seek(0)
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
import seaborn as sns

from commons import makeEnv, saveModel, testModel, trainModel, utils
from agents import agentActorCriticfinal, agentReinforce
from policies import tibNET

SEEDS = [42, 777, 299266, 303489, 295366]

ALGS = [
    "reinforce_baseline",
    "reinforce_standard",
    "reinforce_togo"
] 

MAP = dict.fromkeys(ALGS)

EPISODES_TIMESTEPS = [50000 for _ in range(3)]

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

#PERFORMING EVALUATION FOR 5 DIFFERENT RANDOM SEEDS
for idx in range(5):
    current_seed = source_env.seed(seed=SEEDS[idx])
    for agent, agent_name, duration in zip(MAP, ALGS, EPISODES_TIMESTEPS):
        print(f"Training {agent_name} for {duration} episodes/timesteps. SEED = {current_seed}")

        trainModel.train(
            agent=MAP[agent_name],
            agent_type="reinforce",
            env=source_env,
            actorCriticCheck=False,
            batch_size=0,
            episodes=100,
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
# GENERAZIONE GRAFICI
# for alg in ALGS:
#     seed = SEEDS[2]
#     warhol = utils.Warhol(
#         figure_path=f"./{alg}"
#         )
#     data_reward = np.loadtxt(f"./{alg}/[{seed}]_{alg}_reward_file.txt", delimiter=',', skiprows=1)
#     data_action = np.loadtxt(f"./{alg}/[{seed}]_{alg}_action_file.txt", delimiter=',', skiprows=1)
#     #plot delle reward
#     warhol.plot_column(
#         x=data_reward[:, 2],
#         y=data_reward[:, 1],
#         title=f"Rewards per timestep for {alg}",
#         axis_labels=["timesteps", "reward"],
#         figure_name=f"[{seed}]_{alg}_reward",
#         save=True
#     )
#     #plot delle azioni
#     warhol.plot_column(
#         x=data_action[:, 4],
#         y=data_action[:, 1],
#         title=f"Actiion1 per timestep for {alg}",
#         axis_labels=["timesteps", "action1"],
#         figure_name=f"[{seed}]_{alg}_action1",
#         save=True
#     )
#     warhol.plot_column(
#         x=data_action[:, 4],
#         y=data_action[:, 2],
#         title=f"Action2 per timestep for {alg}",
#         axis_labels=["timesteps", "action2"],
#         figure_name=f"[{seed}]_{alg}_action2",
#         save=True
#     )
#     warhol.plot_column(
#         x=data_action[:, 4],
#         y=data_action[:, 3],
#         title=f"Action3 per timestep for {alg}",
#         axis_labels=["timesteps", "action3"],
#         figure_name=f"[{seed}]_{alg}_action3",
#         save=True
#     )


# seed = SEEDS[2]
# warhol = utils.Warhol(
#     figure_path=f"./"
#     )
# data_reward_baseline = np.loadtxt(f"./reinforce_baseline/[{seed}]_reinforce_baseline_reward_file.txt", delimiter=',', skiprows=1)
# data_reward_standard = np.loadtxt(f"./reinforce_standard/[{seed}]_reinforce_standard_reward_file.txt", delimiter=',', skiprows=1)
# data_reward_togo = np.loadtxt(f"./reinforce_togo/[{seed}]_reinforce_togo_reward_file.txt", delimiter=',', skiprows=1)

# y=[data_reward_standard[:, 1], data_reward_baseline[:, 1], data_reward_togo[:, 1]]

# warhol.plot_columns(
#     y=[data_reward_standard[:, 1], data_reward_baseline[:, 1], data_reward_togo[:, 1]],
#     title=f"Rewards per timestep for various implementations of Reinforce",
#     axis_labels=["timesteps", "rewards"],
#     figure_name=f"[{seed}]_reinforces_rewards",
#     save=True
#     )

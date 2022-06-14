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
from commons.custom_callback import CustomCallback as CB
from env import *
from sb3_contrib.trpo.trpo import TRPO
from stable_baselines3 import PPO

source_env = makeEnv.make_environment("source")
target_env = makeEnv.make_environment("target")
timesteps = 150000

SEEDS = [42, 777, 299266, 303489, 295366]
ALGS = ["ppo", "trpo"]

with open("../step2_tests/best_config.txt", "r") as trpo_best_config_file:
    trpo_best_config = json.load(trpo_best_config_file)

with open("../step2_tests/best_config.txt", "r") as trpo_best_config_file:
    ppo_best_config = json.load(trpo_best_config_file)

if trpo_best_config['configurations']['trpo']['activation_function'] == 'tanh':
    actfunc = torch.nn.Tanh
else:
    actfunc = torch.nn.ReLU

if ppo_best_config['configurations']['ppo']['activation_function'] == 'tanh':
    actfunc = torch.nn.Tanh
else:
    actfunc = torch.nn.ReLU

for idx in range(5):
    print(f"Training PPO and TRPO for {timesteps} timesteps with seed: [{SEEDS[idx]}]")
    current_seed = source_env.seed(seed=SEEDS[idx])

    trpo_callback_ = CB(
        reward_file=f"trpo/{current_seed}_trpo_reward_file.txt",
        action_file=f"trpo/{current_seed}_trpo_action_file.txt"
        )
    ppo_callback_ = CB(
        reward_file=f"ppo/{current_seed}_ppo_reward_file.txt",
        action_file=f"ppo/{current_seed}_ppo_action_file.txt"
        )

    agent_trpo = TRPO(
        env=source_env,
        gamma=trpo_best_config["configurations"]['trpo']['gamma'],
        learning_rate=trpo_best_config["configurations"]['trpo']['lr'],
        policy=trpo_best_config["configurations"]['trpo']['policy'],
        target_kl=trpo_best_config["configurations"]['trpo']['target_kl'],
        policy_kwargs={'activation_fn':actfunc},
        verbose=0,
        seed=SEEDS[idx]
    )

    agent_ppo = PPO(
        env=source_env,
        gamma=ppo_best_config["configurations"]['ppo']['gamma'],
        learning_rate=ppo_best_config["configurations"]['ppo']['lr'],
        policy=ppo_best_config["configurations"]['ppo']['policy'],
        target_kl=ppo_best_config["configurations"]['ppo']['target_kl'],
        policy_kwargs={'activation_fn':actfunc},
        verbose=0,
        seed=SEEDS[idx]
    )

    trainModel.train(
        agent=agent_trpo,
        agent_type="trpo",
        env=source_env,
        callback=trpo_callback_,
        timesteps=timesteps,
        info_file_path="trpo/"
        )

    trainModel.train(
        agent=agent_ppo,
        agent_type="ppo",
        env=source_env,
        callback=ppo_callback_,
        timesteps=timesteps,
        info_file_path="ppo/"
        )

# SEEDLESS RETURNS
for alg in ALGS:

    with open(f"ppo_trpo_sumup/seedless_{alg}_rewards.txt", "w") as seedless_file_header_reward:
        seedless_file_header_reward.write("Episode,Return,Timestep\n")

    for seed in SEEDS:
        with open(f"ppo_trpo_sumup/seedless_{alg}_rewards.txt", "a+") as seedless_file_reward:
            f = open(f"{alg}/{[seed]}_{alg}_reward_file.txt", "r")
            next(f) # salto l'header, non mi interessa averne più di uno.
            seedless_file_reward.write(f.read())
            seedless_file_reward.seek(0)
            f.seek(0)

#SEEDLESS ACTIONS
for alg in ALGS:

    with open(f"ppo_trpo_sumup/seedless_{alg}_actions.txt", "w") as seedless_file_header_action:
        seedless_file_header_action.write("Episode,ActionMeasure\n")

    for seed in SEEDS:
        with open(f"ppo_trpo_sumup/seedless_{alg}_actions.txt", "a+") as seedless_file_action:
            f = open(f"{alg}/{[seed]}_{alg}_action_file.txt", "r")
            next(f) # salto l'header, non mi interessa averne più di uno.
            seedless_file_action.write(f.read())
            seedless_file_action.seek(0)
            f.seek(0)
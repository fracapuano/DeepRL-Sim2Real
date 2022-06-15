import argparse
import json
from pyexpat import model
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from commons import trainModel, testModel, saveModel, makeEnv, utils
from stable_baselines3 import PPO

SEED = 42

with open("ppo/ppo.txt", "r") as ppof:
	ppo_configurations = json.load(ppof)

source_env = makeEnv.make_environment("source")
target_env = makeEnv.make_environment("target")

print(f"Total number of configurations to test {len(ppo_configurations['configurations'])}")

print("Looking for PPO best hyperparameters configuration...")
with open("ppo/ppo_evaluation.txt", "w") as ppo_evaluation_f:
    ppo_evaluation_f.write("ID,ss-return,st-return,tt-return"+'\n')
    for i in range(len(ppo_configurations['configurations'])):
        print(f"Testing configuration ID: {i}")
        config = ppo_configurations['configurations'][i]
        loginfo = [hp for hp in config.items()]
        
        for param in loginfo:
            print(param)

        if config['activation_function'] == 'tanh':
            act_fun = nn.Tanh
        elif config['activation_function'] == 'relu':
            act_fun = nn.ReLU

        agent = PPO(
            policy=config['policy'], 
            env=source_env,
            learning_rate=config['lr'],
            target_kl=config['target_kl'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            seed=SEED,
            policy_kwargs={
            'activation_fn':act_fun
                },
            verbose=0
        )

        agent.learn(total_timesteps=config['timesteps'])
        saveModel.save_model(agent=agent, agent_type='ppo', folder_path='./')
        ss_return, _ = testModel.test(agent, agent_type='ppo', env=source_env, episodes=50, model_info='./ppo-model.mdl', render_bool=False)
        st_return, _ = testModel.test(agent, agent_type='ppo', env=target_env, episodes=50, model_info='./ppo-model.mdl', render_bool=False)

        del agent
        os.remove('ppo-model.mdl')

        agent = PPO(
            policy=config['policy'], 
            env=target_env,
            learning_rate=config['lr'],
            target_kl=config['target_kl'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            seed=SEED,
            policy_kwargs={
                'activation_fn':act_fun
                },
            verbose=0
        )

        agent.learn(total_timesteps=config['timesteps'])
        saveModel.save_model(agent=agent, agent_type='ppo', folder_path='./')
        tt_return, _ = testModel.test(agent, agent_type='ppo', env=target_env, episodes=50, model_info='./ppo-model.mdl', render_bool=False)

        ppo_evaluation_f.write(f"{i},{ss_return},{st_return},{tt_return}"+'\n')
        os.remove('ppo-model.mdl')
ppo_evaluation_f.close()
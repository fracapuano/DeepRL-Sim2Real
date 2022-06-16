import argparse
import json
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
from sb3_contrib.trpo.trpo import TRPO

with open("trpo/trpo.txt", "r") as trpof:
	trpo_configurations = json.load(trpof)

source_env = makeEnv.make_environment("source")
target_env = makeEnv.make_environment("target")

print(f"Total number of configurations to test {len(trpo_configurations['configurations'])}")


print("Looking for TRPO best hyperparameters configuration...")
with open("trpo/trpo_evaluation.txt", "w") as trpo_evaluation_f:
    trpo_evaluation_f.write("ID,ss-return,st-return"+'\n')
    for i in range(len(trpo_configurations['configurations'])):
        print(f"Testing configuration ID: {i}")
        config = trpo_configurations['configurations'][i]
        loginfo = [hp for hp in config.items()]

        for param in loginfo:
            print(param)
        print("Using bounds for m1: " + '\n' + "low: " + str(config['m1_low_bounds']) + '\n' + str(config['m1_high_bounds']))
        print("Using bounds for m2: " + '\n' + "low: " + str(config['m2_low_bounds']) + '\n' + str(config['m2_high_bounds']))
        print("Using bounds for m3: " + '\n' + "low: " + str(config['m3_low_bounds']) + '\n' + str(config['m3_high_bounds']))
        source_env.set_parametrization([config['m1_low_bounds'], config['m1_high_bounds'], config['m2_low_bounds'], config['m2_high_bounds'], config['m3_low_bounds'], config['m3_high_bounds']])
        source_env.set_random_parameters()

        if config['activation_function'] == 'tanh':
            act_fun = nn.Tanh
        elif config['activation_function'] == 'relu':
            act_fun = nn.ReLU

        agent = TRPO(
            policy=config['policy'], 
            env=source_env,
            learning_rate=config['lr'],
            target_kl=config['target_kl'],
            gamma=config['gamma'],
            batch_size=config['batch_size'],
            policy_kwargs={
            'activation_fn':act_fun
                },
            verbose=0
        )

        agent.learn(total_timesteps=config['timesteps'])
        saveModel.save_model(agent=agent, agent_type='trpo', folder_path='./')
        ss_return, _ = testModel.test(agent, agent_type='trpo', env=source_env, episodes=50, model_info='./trpo-model.mdl', render_bool=False)
        st_return, _ = testModel.test(agent, agent_type='trpo', env=target_env, episodes=50, model_info='./trpo-model.mdl', render_bool=False)

        del agent
        os.remove('trpo-model.mdl')

        trpo_evaluation_f.write(f"{i},{ss_return},{st_return}"+'\n')
trpo_evaluation_f.close()



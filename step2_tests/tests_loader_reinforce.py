"""
Performing training and testing of REINFORCE over source and target domain to retrive source-source return, source-target return and target-target return.
Each time the REINFORCE agent is initialized with a different set of hyperparameters taken from reinforce/reinforce.txt.
The obtained results are stored inside reinforce_evaluation.txt.
"""

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
from policies import fraNET, tibNET

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--print-every', default=1, type=int, help='number of episodes to pass before printing an output')

    return parser.parse_args()

args = parse_args()

with open("reinforce/reinforce.txt", "r") as rf:
	reinforce_configurations = json.load(rf)

source_env = makeEnv.make_environment("source")
target_env = makeEnv.make_environment("target")

observation_space_dim = source_env.observation_space.shape[-1]
action_space_dim = source_env.action_space.shape[-1]

#REINFORCE OPTIMIZATION
print("Looking for REINFORCE best hyperparameters configuration...")
with open("reinforce/reinforce_evaluation.txt", "w") as reinforce_evaluation_f:
	reinforce_evaluation_f.write("ID,ss-return,st-return,tt-return"+'\n')
	for i in tqdm(range(len(reinforce_configurations['configurations']))):
		config = reinforce_configurations['configurations'][i]
		loginfo = [hp for hp in config.items()]
		print("Testing ")
		for param in loginfo:
			print(param)
		import agents.agentReinforce

		policy = tibNET.ReinforcePolicy(
    		state_space=observation_space_dim,
    		action_space=action_space_dim,
    		hidden=config['n_neurons'],
    		init_sigma=config['sigma']
    	)

		agent = agents.agentReinforce.Agent(
    		policy,
    		device='cpu',
    		gamma=config['gamma'],
    		lr=config['lr']
    	)

		trainModel.train(
			agent,
			"reinforce",
			source_env, 
			actorCriticCheck=False, 
			batch_size=config['batch_size'], 
			episodes=config['n_episodes'],
			print_every=args.print_every,
			save_to_file_bool=False
			)

		ss_return, _ = testModel.test(agent, agent_type='reinforce', env=source_env, episodes=50, render_bool=False)
		st_return, _ = testModel.test(agent, agent_type='reinforce', env=target_env, episodes=50, render_bool=False)

		del policy
		del agent

		policy = tibNET.ReinforcePolicy(
    		state_space=observation_space_dim,
    		action_space=action_space_dim,
    		hidden=config['n_neurons'],
    		init_sigma=config['sigma']
    		)
		agent = agents.agentReinforce.Agent(
    		policy,
    		device='cpu',
    		gamma=config['gamma'],
    		lr=config['lr']
    		)

		trainModel.train(agent,
			"reinforce",
			target_env, 
			actorCriticCheck=False, 
			batch_size=config['batch_size'], 
			episodes=config['n_episodes'],
			print_every=args.print_every,
			save_to_file_bool=False
			)

		tt_return, _ = testModel.test(agent, agent_type='reinforce', env=target_env, episodes=50, render_bool=False)

		reinforce_evaluation_f.write(f"{i},{ss_return},{st_return},{tt_return}"+'\n')
reinforce_evaluation_f.close()
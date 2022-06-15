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

with open("a2c/actorCritic.txt", "r") as acf:
	actorCritic_configurations = json.load(acf)

source_env = makeEnv.make_environment("source")
target_env = makeEnv.make_environment("target")

observation_space_dim = source_env.observation_space.shape[-1]
action_space_dim = source_env.action_space.shape[-1]

#ACTOR-CRITIC OPTIMIZATION
print("Looking for ActorCritic best hyperparameters configuration...")
with open("a2c/actorCritic_evaluation.txt", "w") as actorCritic_evaluation_f:
	actorCritic_evaluation_f.write("ID,ss-return,st-return,tt-return"+'\n')
	for i in tqdm(range(len(actorCritic_configurations['configurations']))):
		config = actorCritic_configurations['configurations'][i]
		loginfo = [hp for hp in config.items()]
		print("Testing ")
		for param in loginfo:
			print(param)
		import agents.agentActorCritic

		policy = tibNET.ActorCriticPolicy(
    		state_space=observation_space_dim,
    		action_space=action_space_dim,
    		hidden=config['n_neurons'],
    		init_sigma=config['sigma']
    	)

		agent = agents.agentActorCritic.Agent(
    		policy,
			net_type='tibNET',
    		device='cpu',
    		gamma=config['gamma'],
    		lr=config['lr']
    	)

		trainModel.train(
			agent,
			"actorcritic",
			source_env, 
			actorCriticCheck=False, 
			batch_size=config['batch_size'], 
			episodes=config['n_episodes'],
			print_every=args.print_every
			)

		ss_return, _ = testModel.test(agent, agent_type='actorCritic', env=source_env, episodes=50, render_bool=False)
		print("Configuration source-source return: ", ss_return)
		st_return, _ = testModel.test(agent, agent_type='actorCritic', env=target_env, episodes=50, render_bool=False)
		print("Configuration source-target return: ", st_return)

		del policy
		del agent

		policy = tibNET.ActorCriticPolicy(
    		state_space=observation_space_dim,
    		action_space=action_space_dim,
    		hidden=config['n_neurons'],
    		init_sigma=config['sigma']
    		)
			
		agent = agents.agentActorCritic.Agent(
    		policy,
    		device='cpu',
    		gamma=config['gamma'],
			net_type='tibNET',
    		lr=config['lr']
    		)

		trainModel.train(agent,
			"actorcritic", 
			target_env, 
			actorCriticCheck=False, 
			batch_size=config['batch_size'], 
			episodes=config['n_episodes'], 
			print_every=args.print_every
			)

		tt_return, _ = testModel.test(agent, agent_type='actorCritic', env=target_env, episodes=50, render_bool=False)

		actorCritic_evaluation_f.write(f"{i},{ss_return},{st_return},{tt_return}"+'\n')
actorCritic_evaluation_f.close()

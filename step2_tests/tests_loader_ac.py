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
from policies import fraNET

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
	actorCritic_evaluation_f.write("ID,ss-return,st-return"+'\n')
	for i in tqdm(range(len(actorCritic_configurations['configurations']))):
		config = actorCritic_configurations['configurations'][i]
		loginfo = [hp for hp in config.items()]
		print("Testing ")
		for param in loginfo:
			print(param)
		import agents.agentActorCriticfinal

		if config['activation_function'] == 'tanh':
			act_fun = np.array([nn.Tanh for _ in range(2)])
		elif config['activation_function'] == 'relu':
			act_fun = np.array([nn.ReLU for _ in range(2)])

		policy = fraNET.ActorCriticPolicy(
    		state_space=observation_space_dim,
    		action_space=action_space_dim,
    		#hidden_layers=config['n_layers'],
    		#hidden_neurons=np.array([config['n_neurons'] for _ in range(config['n_layers'])]),
    		activation_function=act_fun,
    		init_sigma=config['sigma']
    	)

		agent = agents.agentActorCriticfinal.Agent(
    		policy,
			net_type='fraNET',
    		device='cpu',
    		gamma=config['gamma'],
    		lr=config['lr']
    	)

		trainModel.train(
			agent,
			source_env, 
			actorCriticCheck=False, 
			batch_size=config['batch_size'], 
			episodes=config['n_episodes'], 
			print_every=args.print_every
			)

		ss_return = testModel.test(agent, agent_type='actorCritic', env=source_env, episodes=50, render_bool=False)
		st_return = testModel.test(agent, agent_type='actorCritic', env=target_env, episodes=50, render_bool=False)

		del policy
		del agent

		policy = fraNET.ActorCriticPolicy(
    		state_space=observation_space_dim,
    		action_space=action_space_dim,
    		#hidden_layers=config['n_layers'],
    		#hidden_neurons=np.array([config['n_neurons'] for _ in range(config['n_layers'])]),
    		activation_function=act_fun,
    		init_sigma=config['sigma']
    		)
		agent = agents.agentActorCriticfinal.Agent(
    		policy,
    		device='cpu',
    		gamma=config['gamma'],
			net_type='fraNET',
    		lr=config['lr']
    		)

		trainModel.train(agent, 
			target_env, 
			actorCriticCheck=False, 
			batch_size=config['batch_size'], 
			episodes=config['n_episodes'], 
			print_every=args.print_every
			)

		tt_return = testModel.test(agent, agent_type='actorCritic', env=target_env, episodes=50, render_bool=False)

		actorCritic_evaluation_f.write(f"{i},{ss_return},{st_return},{tt_return}"+'\n')
actorCritic_evaluation_f.close()

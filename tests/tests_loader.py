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
    parser.add_argument('--print-every', default=10, type=int, help='number of episodes to pass before printing an output')

    return parser.parse_args()

args = parse_args()

reinforce_configurations = json.load("reinforce.txt")
actorCritic_configurations = json.load("actorCritic.txt")
# PPO
# TRPO

source_env = makeEnv.make_environment("source")
target_env = makeEnv.make_environment("target")

observation_space_dim = source_env.observation_space.shape[-1]
action_space_dim = source_env.action_space.shape[-1]

#REINFORCE OPTIMIZATION
for i in tqdm(range(len(reinforce_configurations['configurations']))):
	config = reinforce_configurations['configurations'][i]
	import agents.agentReinforce

	if config['activation_function'] == 'tanh':
		act_fun = np.array([nn.Tanh for _ in range(config['n_layers'])])
	elif config['activation_function'] == 'relu':
		act_fun = np.array([nn.ReLU for _ in range(config['n_layers'])])

    policy = fraNET.ReinforcePolicy(
    	state_space=observation_space_dim,
    	action_space=action_space_dim,
    	hidden_layers=config['n_layers'],
    	hidden_neurons=np.array([config['n_neurons'] for _ in range(config['n_layers'])]),
    	activation_function=act_fun,
    	init_sigm=config['sigma']
    )
    agent = agents.agentReinforce.Agent(
    	policy,
    	device='cpu',
    	gamma=config['gamma'],
    	lr=config['lr']
    )

    trainModel.train(agent, source_env, actorCriticCheck=False, config['batch_size'], config['n_episodes'], args.print_every)

    ss_return = testMode.test(agent, agent_type='reinforce', env=source_env, episodes=50, render_bool=False)
    st_return = testMode.test(agent, agent_type='reinforce', env=target_env, episodes=50, render_bool=False)

    del policy
    del agent

    policy = ReinforcePolicy(
    	state_space=observation_space_dim,
    	action_space=action_space_dim,
    	hidden_layers=config['n_layers'],
    	hidden_neurons=np.array([config['n_neurons'] for _ in range(config['n_layers'])]),
    	activation_function=act_fun,
    	init_sigm=config['sigma']
    	)
    agent = agents.agentReinforce.Agent(/
    	policy,
    	device='cpu',
    	gamma=config['gamma'],
    	lr=config['lr']
    	)

    trainModel.train(agent, target_env, actorCriticCheck=False, config['batch_size'], config['n_episodes'], args.print_every)
    tt_return = testMode.test(agent, agent_type='reinforce', env=target_env, episodes=50, render_bool=False)

    with open("reinforce_evaluation.txt", "a") as f:
    	f.write(f"{i},{ss_return},{st_return},{tt_return}")

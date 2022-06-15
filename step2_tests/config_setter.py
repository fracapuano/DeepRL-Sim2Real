import json
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

def write_config(FILE_NAME, configurations):
	configuration={'configurations':[]}
	with open(FILE_NAME, "w") as f:
		for config in tqdm(configurations):
			configuration['configurations'].append(config)
		json.dump(configuration, f, indent=4)
		print(f"Total number of configurations: {len(configuration['configurations'])}")

REINFORCE_PARAMS = {
	'n_neurons':[32, 64],
	'gamma':[0.998, 0.999],
	'lr':[1e-3, 5e-3],
	'n_episodes':[50000],
	'sigma':[0.25, 0.5, 1.25],
	'batch_size':[0]
}

ACTOR_CRITIC_PARAMS = {
	'n_neurons':[64],
	'gamma':[0.998, 0.999],
	'lr':[1e-3, 1e-2],
	'n_episodes':[20000],
	'batch_size':[5, 10],
	'sigma':[0.5]
}

TRPO_PARAMS = {
	'policy':['MlpPolicy'],
	'lr':[1e-3, 5e-3],
	'gamma':[0.998, 0.999],
	'target_kl':[0.01, 0.05],
	'timesteps':[150000],
	'activation_function':['tanh'],
	'batch_size':[128, 256]
}

PPO_PARAMS = {
	'policy':['MlpPolicy'],
	'lr':[1e-3, 2.25e-3, 5e-3],
	'gamma':[0.998, 0.999],
	'target_kl':[0.01, 0.05],
	'timesteps':[150000],
	'activation_function':['tanh'],
	'batch_size':[128, 256]
}

write_config("reinforce/reinforce.txt", ParameterGrid(REINFORCE_PARAMS))
write_config("a2c/actorCritic.txt", ParameterGrid(ACTOR_CRITIC_PARAMS))
write_config("ppo/ppo.txt", ParameterGrid(PPO_PARAMS))
write_config("trpo/trpo.txt", ParameterGrid(TRPO_PARAMS))
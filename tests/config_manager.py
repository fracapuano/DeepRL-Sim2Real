import json
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
from tqdm import tqdm

def write_config(FILE_NAME, configurations):
	configuration={'configurations':[]}
	with open(FILE_NAME, "w") as f:
		for config in tqdm(configurations):
			configuration['configurations'].append(config)
		json.dump(configuration, f, indent=4)
		print(f"Total number of configurations: {len(configuration['configurations'])}")

REINFORCE_PARAMS = {
	#'n_neurons':[32, 64],
	#'n_layers':[2, 4],
	'gamma':[0.998, 0.999],
	'lr':[1e-3, 5e-3],
	'activation_function':['tanh', 'relu'],
	'n_episodes':[100000, 200000],
	'sigma':[0.25, 0.5, 0.75],
	'batch_size':[0]
}

ACTOR_CRITIC_PARAMS = {
	#'n_neurons':[32, 64],
	#'n_layers':[2, 4],
	'gamma':[0.998, 0.999],
	'lr':[1e-3, 5e-3],
	'activation_function':['tanh', 'relu'],
	'n_episodes':[100000, 200000],
	'batch_size':[20, 50],
	'sigma':[0.25, 0.5, 0.75]
}

write_config("reinforce.txt", ParameterGrid(REINFORCE_PARAMS))
write_config("actorCritic.txt", ParameterGrid(ACTOR_CRITIC_PARAMS))



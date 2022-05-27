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

TRPO_PARAMS = {
	'policy':['MlpPolicy'],
	'lr':[1e-3, 5e-3],
	'gamma':[0.998, 0.999],
	'target_kl':[0.001, 0.01, 0.05],
	'episodes':[100000],
	'activation_function':['tanh'],
	#'use_sde':[True, False],
	#'batch_size':[128, 256],
	'low_bounds':[2, 4, 5],
	'high_bounds':[4.5, 5.5, 7]
}

write_config("trpo/trpo.txt", ParameterGrid(TRPO_PARAMS))



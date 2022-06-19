"""
This script retrives the best hyperparameter's configuration obtained at the end of the tests.
"""

import pandas as pd
import numpy as np
import json

models = [
    'reinforce',
    'actorCritic',
    'ppo',
    'trpo'
]

best_configurations = {'configurations': dict.fromkeys([f"{model}" for model in models], {})}

with open("best_config.txt", "w") as bc:

    for model in models:

        config_df = pd.read_csv(f"{model}/{model}_evaluation.txt")

        with open(f"{model}/{model}.txt", "r") as configs_file:
            configs = json.load(configs_file)

            best_configurations['configurations'][f"{model}"]=configs['configurations'][config_df['st-return'].idxmax()]

    json.dump(best_configurations, bc, indent=4)
import pandas as pd
import json


best_configurations = {'best_config': {}}

with open("best_config.txt", "w") as bc:
    config_df = pd.read_csv("trpo/trpo_evaluation.txt")

    with open("trpo/trpo.txt", "r") as configs_file:
        configs = json.load(configs_file)

        best_configurations['best_config'] = configs['configurations'][config_df['st-return'].idxmax()]

    json.dump(best_configurations, bc, indent=4)
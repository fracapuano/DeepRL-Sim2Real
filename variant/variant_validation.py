import numpy as np
import torch.nn as nn
import json

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from commons import trainModel, testModel, saveModel, makeEnv, utils
from adaptive_dr.BayRn import get_bc
from sb3_contrib.trpo import TRPO
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-timesteps', default=100000, type=int, help='Number of timesteps to train the agent on')
    parser.add_argument('--test-episodes', default=50, type=int, help='Number of timesteps to test the agent in the target environment')
    return parser.parse_args()

args = parse_args()

def main(): 
    # randomizing only "leg" and "foot" considering the output of dynamics_analysis.py
    bounds = get_bc(masses = ["leg", "foot"])
    

    # INIZIALIZZO TRPO CON LA MIGLIOR CONFIGURAZIONE TROVATA IN STEP4
    with open("./step4_tests/best_config.txt", "r") as trpo_best_config_file:
        best_config = json.load(trpo_best_config_file)

    actfunc = nn.Tanh
    source_env, target_env = makeEnv.make_environment('source'), makeEnv.make_environment('target')

    source_env.set_parametrization(bounds)
    source_env.set_random_parameters()

    agent = TRPO(
        env=source_env,
        gamma=best_config['best_config']['gamma'],
        learning_rate=best_config['best_config']['lr'],
        policy=best_config['best_config']['policy'],
        target_kl=best_config['best_config']['target_kl'],
        policy_kwargs={'activation_fn':actfunc},
        verbose=1
    )

    # TRAIN TRPO_BEST_CONFIG SU SOURCE USANDO I BOUNDS DI GET_BC() DI BAYRN
    print(f"Testing over {args.train_timesteps}...")
    agent.learn(total_timesteps=args.train_timesteps)
    saveModel.save_model(agent=agent, agent_type='trpo', folder_path='./')

    # TEST TRPO_BEST_CONFIG SU TARGET
    st_return = testModel.test(agent, agent_type='trpo', env=target_env, episodes=args.test_timesteps, model_info='./trpo-model.mdl', render_bool=True)
    print(f"Current Source-Target return over {args.test_timesteps} test episodes: {st_return}")

if __name__ == "__main__": 
    main()
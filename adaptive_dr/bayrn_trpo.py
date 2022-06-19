import argparse
import json
import string
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
from sb3_contrib.trpo.trpo import TRPO
from BayRn import get_bc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-timesteps', default=250000, type=int, help='Number of timesteps to train the agent on')
    parser.add_argument('--test-timesteps', default=50, type=int, help='Number of timesteps to test the agent in the target environment')
    return parser.parse_args()

args = parse_args()

print("Optimizing distrubtions' bounds with BayRn...")
bounds = get_bc()
print(f"Using bounds: {bounds}")

source_env, target_env = makeEnv.make_environment('source'), makeEnv.make_environment('target')

source_env.set_parametrization(bounds)
source_env.set_random_parameters()

agent = TRPO(env=source_env, policy="MlpPolicy", verbose=1)
    # TRAIN TRPO_BEST_CONFIG SU SOURCE USANDO I BOUNDS DI GET_BC() DI BAYRN
print(f"Testing over {args.train_timesteps} timesteps...")
agent.learn(total_timesteps=args.train_timesteps)
saveModel.save_model(agent=agent, agent_type='trpo', folder_path='./')

# TEST TRPO_BEST_CONFIG SU TARGET
ss_return, ss_std = testModel.test(agent, agent_type='trpo', env=source_env, episodes=args.test_timesteps, model_info='./trpo-model.mdl', render_bool=False)
st_return, st_std = testModel.test(agent, agent_type='trpo', env=target_env, episodes=args.test_timesteps, model_info='./trpo-model.mdl', render_bool=False)
print(f"Current Source-Source return over {args.test_timesteps} test episodes: {ss_return} pm {ss_std}")
print(f"Current Source-Target return over {args.test_timesteps} test episodes: {st_return} pm {st_std}")
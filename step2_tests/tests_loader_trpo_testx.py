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
from sb3_contrib.trpo.trpo import TRPO

source_env = makeEnv.make_environment("source")
target_env = makeEnv.make_environment("target")

with open("trpo/trpo_evaluation.txt", "w") as trpo_evaluation_f:
    trpo_evaluation_f.write("ss-return,st-return,tt-return"+'\n')
    agent = TRPO(policy='MlpPolicy', env=source_env, verbose=1)
    agent.learn(total_timesteps=50000)
    saveModel.save_model(agent=agent, agent_type='trpo', folder_path='./')
    ss_return = testModel.test(agent, agent_type='trpo', env=source_env, episodes=50, model_info='./trpo-model.mdl', render_bool=False)
    st_return = testModel.test(agent, agent_type='trpo', env=target_env, episodes=50, model_info='./trpo-model.mdl', render_bool=False)

    del agent
    os.remove('trpo-model.mdl')

    agent = TRPO(policy='MlpPolicy', env=target_env, verbose=1)
    agent.learn(total_timesteps=50000)
    saveModel.save_model(agent=agent, agent_type='trpo', folder_path='./')
    tt_return = testModel.test(agent, agent_type='trpo', env=target_env, episodes=50, model_info='./trpo-model.mdl', render_bool=False)

    trpo_evaluation_f.write(f"{ss_return},{st_return},{tt_return}"+'\n')
    os.remove('trpo-model.mdl')
trpo_evaluation_f.close()



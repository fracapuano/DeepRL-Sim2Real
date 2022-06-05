"""
This script uses domain randomization to effectively train a RL agent with a variety of possible masses
so to make the learned policy robust to changes in the overall distribution.
The used algorithm is the Trust Region Policy Optimization (TRPO) algorithm, considering that in the "stable" training
phase it has been shown to be the most promising one. 
"""
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from sb3_contrib.trpo.trpo import TRPO
from env.custom_hopper import *
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--domain-type', default='source', type=str, help='source / target')
    parser.add_argument('--low', default=0, type=float, help='lower bound to uniform distribution')
    parser.add_argument('--high', default=5, type=float, help='upper bound to uniform distribution')
    parser.add_argument('--timesteps', default=2048, type=int, help='Timesteps used to train the agent')
    return parser.parse_args()

args = parse_args()

def main():
    if args.domain_type.lower() == 'source':
        env = gym.make("CustomHopper-source-v0")
    if args.domain_type.lower() == 'target': 
        env = gym.make("CustomHopper-target-v0")
        
    # instantiating an agent
    agent = TRPO('MlpPolicy', env)

    default_bounds = [args.low, args.high, args.low, args.high, args.low, args.high]
    env.set_parametrization(default_bounds)

    episodes = tqdm(range(args.n_episodes))

    for episode in episodes: 
        episodes.set_description(f"Episode {episode}")

        # changing the masses of the hopper
        env.set_random_parameters()

        # learning with respect to random environment considered
        agent.learn(total_timesteps = args.timesteps)

    agent.save(f"udr_trpo_unif{str(int(args.low))}-{str(int(args.high))}.mdl")   

if __name__ == '__main__':
    main()
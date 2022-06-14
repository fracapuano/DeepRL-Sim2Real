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
    parser.add_argument('--n-episodes', default=50, type=int, help='Number of training episodes')
    parser.add_argument('--domain-type', default='source', type=str, help='source / target')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--model', default=None, type=str, help='Trained agent to import')
    parser.add_argument('--timesteps', default=2048, type=int, help='Timesteps used to train the agent')
    return parser.parse_args()

args = parse_args()

def main():
    if args.domain_type.lower() == 'source':
        env = gym.make("CustomHopper-source-v0")
    if args.domain_type.lower() == 'target': 
        env = gym.make("CustomHopper-target-v0")
    
    # instantiating an agent that reads the model collected in traning
    agent = TRPO.load(args.model)
    episodes = tqdm(range(args.n_episodes))
    #MAX_TIMESTEP = 500
    GAMMA = 0.99

    obs = env.reset()
    results = np.zeros(args.n_episodes)

    for episode in episodes: 
        episodes.set_description(f"Episode {episode}")

        done = False
        test_rewards = np.zeros(args.timesteps)
        obs = env.reset()
        timestep = 0

        while not done: 
            # using the policy to select an action in the current state
            action, _ = agent.predict(obs)
            # stepping the environment with respect to the action selected
            obs, rewards, done, _ = env.step(action)

            test_rewards[timestep] = rewards
            timestep += 1

            if args.render:
                env.render()

        results[episode] = test_rewards.sum()

    print("*** Average total reward over episodes : {:.4f} ***".format(results.mean()))
    print("*** Average reward Standard Deviation over episodes : {:.4f} ***".format(results.std()))


if __name__ == '__main__':
    main()
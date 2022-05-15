"""
Implementing Bayesian Optimisation
"""
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


from torch.distributions.uniform import Uniform
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from sb3_contrib.trpo.trpo import TRPO
from env.custom_hopper import *
import argparse
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    # number of interaction with target environment: the higher, the better, with the contraint
    # of not having too many since after a certain moment it would train on the target environment
    parser.add_argument('--n-roll', default=3, type=int, help='Number of rollout on the target environment')

    parser.add_argument('--min', default=0.5, type=float, help='lower bound to masses distribution')
    parser.add_argument('--max', default=25, type=float, help='upper bound to masses distribution')

    parser.add_argument('--success reward', default=2000, type=float, help='Reward after which declaring convergence for the bayes optimization')
    parser.add_argument('--maxit', default=100, type=int, help = 'Maximal number of iterations for Bayesian Optimization')
    return parser.parse_args()
args = parse_args()


def blackbox_fun(**kwargs): 
    """
    Random scalar function that, in the region [0,8] has one maximum only
    """
    x = np.fromiter(kwargs.values(), dtype=float)
    # if x[0] <= x[1]: 
    #     massesDistribution = Uniform(low = torch.tensor([x[0]], dtype = float), high = torch.tensor([x[1]], dtype = float))
    # else: 
    #     massesDistribution = Uniform(low = torch.tensor([x[1]], dtype = float), high = torch.tensor([x[0]], dtype = float))

    # m0 = massesDistribution.sample().detach().numpy().item()
    # m1 = massesDistribution.sample().detach().numpy().item()
    m0, m1 = x[0], x[1]

    output = - m1 ** 2 + np.sin(m0)
    return output

def J_masses(**kwargs):
    # lower (x[0]) and upper (x[1]) bound of the distribution
    x = np.fromiter(kwargs.values(), dtype=float)

    # create the source and target environments
    GAMMA = 0.99
    source_env = gym.make("CustomHopper-source-v0")
    target_env = gym.make("CustomHopper-target-v0") 

    # setting as parameters of the distribution the current elements in x
    if x[1] >= x[0]:
        source_env.env.low = x[0]
        source_env.env.high = x[1]
    else: 
        source_env.env.low = x[1]
        source_env.env.high = x[0]
    
    # sampling with respect to the parameters just passed to set random masses
    source_env.set_random_parameters()

    # istantiating an agent
    agent = TRPO('MlpPolicy', source_env)
    # learning with respect to random environment considered
    agent.learn(total_timesteps = 2048)

    roll_return = []
    # testing the learned policy in the target environment for n_roll times
    for rollout in range(args.n_roll): 
        done = False
        test_rewards = []
        obs = target_env.reset()
        timestep = 0
        while not done: 
            # using the policy to select an action in the current state
            action, _ = agent.predict(obs)
            # stepping the environment with respect to the action selected
            obs, rewards, done, _ = target_env.step(action)
            # collecting the reward (to later obtain return)
            test_rewards.append(rewards)

            timestep += 1

        gammas = GAMMA ** np.arange(timestep)
        roll_return.append(test_rewards @ gammas)
    
    roll_return = np.array(roll_return)
    return roll_return.mean()

def main():
    
    pbounds = {'low': (args.min, args.max), 'high': (args.min, args.max)}
    
    bounds_transformer = SequentialDomainReductionTransformer()

    mutating_optimizer = BayesianOptimization(
    f=J_masses,
    pbounds=pbounds,
    verbose=1,
    random_state=1,
    bounds_transformer=bounds_transformer
    )

    mutating_optimizer.maximize(
    init_points=2,
    n_iter=args.maxit,
    )

    print(mutating_optimizer.max)

if __name__ == '__main__':
    main()
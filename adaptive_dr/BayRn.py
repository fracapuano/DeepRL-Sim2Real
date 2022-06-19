import sys
import os
import inspect
import argparse
import numpy as np
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import torch
from torch.distributions.uniform import Uniform
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from sb3_contrib.trpo.trpo import TRPO
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()

    # number of interaction with target environment: the higher, the better, with the contraint
    # of not having too many since after a certain moment it would train on the target environment
    parser.add_argument('--n-roll', default=5, type=int, help='Number of rollout on the target environment')

    parser.add_argument('--min', default=0.25, type=float, help='lower bound to masses distribution')
    parser.add_argument('--max', default=10, type=float, help='upper bound to masses distribution')

    parser.add_argument('--n-init', default=5, type=int, help='Number of initialization iterations')
    parser.add_argument('--maxit', default=15, type=int, help = 'Maximal number of iterations for Bayesian Optimization')
    parser.add_argument('--timesteps', default=int(1e5), type=int, help='Number of timesteps for policy algorithm')

    parser.add_argument('--n-samples', default=5, type=int, help="Number of samples to evaluate distribution")
    return parser.parse_args()

args = parse_args()

def init_D(agent, source_env, target_env, masses, n_init = args.n_init, n_roll = args.n_roll):
    """
    This function returns an initial dataset D in which each row is D[row, :-1], D[row, -1] = target, parameters
    Parameters: 
        n_init: number of initialization steps to be considered
        n_roll: number of rollout to obtain an estimate of the actual value of the unknown objective function
    Returns: 
        D: dataset of parameters and objective function values
    """

    low = args.min
    high = args.max
    ncols = 6

    parametersDistribution = Uniform(low = torch.tensor([low], dtype = float), 
                                 high = torch.tensor([high], dtype = float))

    D = torch.zeros(n_init, ncols+1)
    
    for i in range(n_init): 
        phi_i = torch.tensor([parametersDistribution.sample() for _ in range(ncols)], dtype = float)
        D[i, :-1] = phi_i
        D[i, -1] = J_masses(agent=agent, source_env=source_env, target_env=target_env, bounds=phi_i, masses=masses)
        
    return D

def J_masses(agent, source_env, target_env, bounds, masses, n_samples=args.n_samples):
    # sampling with respect to the parameters just passed to set random masses
    # for n in range(n_samples):
    #     source_env.set_parametrization(bounds)
    #     source_env.set_random_parameters(masses = masses, dist_type="uniform")

    #     # learning with respect to random environment considered
    #     agent.learn(total_timesteps = args.timesteps)

    source_env.set_parametrization(bounds)
    source_env.set_random_parameters(masses = masses, dist_type="uniform")

    # learning with respect to random environment considered
    agent.learn(total_timesteps = args.timesteps)

    roll_return = []
    # testing the learned policy in the target environment for n_roll times
    for rollout in range(args.n_roll): 
        done = False
        test_rewards = []
        obs = target_env.reset()
        while not done: 
            # using the policy to select an action in the current state
            action, _ = agent.predict(obs)
            # stepping the environment with respect to the action selected
            obs, rewards, done, _ = target_env.step(action)
            # collecting the reward
            test_rewards.append(rewards)


        # obtaining total return
        roll_return.append(np.array(test_rewards).sum())
    
    roll_return = np.array(roll_return)
    return roll_return.mean()

def obtain_candidate(X, Y): 
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    UCB = UpperConfidenceBound(gp, beta=0.1, maximize = True)

    bounds = torch.stack([args.min * torch.ones(X.shape[1]), args.max * torch.ones(X.shape[1])])
    
    candidate, _ = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
    
    candidate = candidate.reshape(-1,)
    return candidate

def BayRN(masses = ["tigh", "leg", "foot"], n_init = args.n_init, n_roll = args.n_roll, maxit = args.maxit, verbose = 1): 
    """
    This function uses bayesian optimization to choose the optimal parametrization given a specific set of parameters
    for what concerns their influence on a some blackbox function. 
    Parameters: 
        n_init: number of initializations iterations to be run so to collect some initial evidence. 
                during such iterations everything is completely random
        n_roll: number of rollout iterations to use so to estimate the actual outcome of some specific set of 
                parameters
        maxit: maximal number of iterations of the overall Bayesian Optimization process. 
    """

    # creating source and target environments
    source_env = gym.make("CustomHopper-source-v0")
    print("Initial masses: ", source_env.sim.model.body_mass)
    target_env = gym.make("CustomHopper-target-v0") 

    # istantiating an agent
    agent = TRPO('MlpPolicy', source_env, verbose=verbose)

    D = init_D(agent, source_env, target_env, masses, n_init = n_init, n_roll = n_roll)
    
    for it in tqdm(range(maxit)):  
        X, Y = D[:, :-1], D[:, -1].reshape(-1,1)
        # obtaining best candidate with Bayesian Optimization
        candidate = obtain_candidate(X, Y)
        # evaluating the candidate solution with source training - rollout evaluation
        J_phi = J_masses(agent, source_env, target_env, candidate, masses)
        J_phi = torch.tensor(J_phi).reshape(-1)

        candidate_and_J = torch.hstack([candidate, J_phi])
        
        D = torch.vstack(
            (D, candidate_and_J)
        )
    
    bestCandidate = D[torch.argmax(D[:, -1]), :-1]
    return D, bestCandidate

def get_bc(masses=["tigh", "leg", "foot"], verbose=0):
    D, bc = BayRN(masses=masses, verbose=verbose)
    np.savetxt("BayRN_D.txt", D)
    return bc

def main():
    print(get_bc())

if __name__ == '__main__':
    main()
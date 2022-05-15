"""
Implementing BayRn [13] paper to carry out Bayesian Optimization of the distribution parameters.
"""
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from sb3_contrib.trpo.trpo import TRPO
from env.custom_hopper import *
from torch.distributions.distribution import Uniform
import argparse
import numpy as np
from tqdm import tqdm
    
"""
Since no observations are available on the noise level
"""
from botorch.models import SingleTaskGP

from botorch.fit import fit_gpytorch_model
"""
Since the inputs have to be standardized
"""
from botorch.utils import standardize

from gpytorch.mlls import ExactMarginalLogLikelihood
"""
For the considered acquisition function
"""
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-init', default=3, type=int, help='Number of initialization policies')

    # number of interaction with target environment: the higher, the better, with the contraint
    # of not having too many since after a certain moment it would train on the target environment
    parser.add_argument('--n-roll', default=3, type=int, help='Number of rollout on the target environment')

    parser.add_argument('--min', default=0.5, type=float, help='lower bound to masses distribution')
    parser.add_argument('--max', default=25, type=float, help='upper bound to masses distribution')

    parser.add_argument('--success reward', default=2000, type=float, help='Reward after which declaring convergence for the bayes optimization')
    parser.add_argument('--maxit', default=100, type=int, help = 'Maximal number of iterations for Bayesian Optimization')
    return parser.parse_args()
args = parse_args()
def main():
    
    source_env = gym.make("CustomHopper-source-v0")
    target_env = gym.make("CustomHopper-target-v0")
    GAMMA = 0.99

    """
    The overall optimization starts from considering a Uniform distribution having bound as [min, max]
    """
    massesDistribution = Uniform(low = torch.tensor([args.min], dtype = float), high = torch.tensor([args.max], dtype = float))
    
    """
        INITIALIZATION STEP
    """
    """
    Since the distribution considered is a Uniform Distribution, it is parametrized with the lower and upper bound only, therefore
    the number of parameters is equal to 2. 
    """
    n_parameters = 2    
    phi_init = np.zeros(n_parameters, args.n_init)
    return_on_target = np.zeros(args.n_roll)

    """
    Populating each column of phi_init with lower (first) and upper (second) bound considered for the distribution
    sampled from the distribution having [min, max] as parameters
    """
    agent = TRPO('MlpPolicy', source_env)

    for init in range(args.n_init): 
        col_init = np.array([massesDistribution.sample().detach().numpy(), massesDistribution.sample().detach().numpy()])
        phi_init[:, init] = col_init

        # setting as parameters of the distribution the current elements in col_init
        source_env.env.low = col_init[0]
        source_env.env.high = col_init[1]

        # sampling with respect to the parameters just passed
        source_env.set_random_parameters()

        # learning with respect to random environment considered
        agent.learn(verbose = 1)

        # testing the learned policy in the target environment for n_roll times
        roll_return = np.array([])
        for rollout in range(args.n_roll): 
            done = False
            test_rewards = np.array([])
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
            roll_return[rollout] = test_rewards @ gammas
        
        print(roll_return.mean())
        return_on_target[init] = roll_return.mean()

    
    return_on_target = standardize(return_on_target)

    # tensorifying Gaussian Process inputs
    return_on_target, phi_init = torch.tensor(return_on_target), torch.tensor(phi_init)
    # fitting a Gaussian Process model
    gp = SingleTaskGP(return_on_target, phi_init)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    # building an acquisition function
    UCB = UpperConfidenceBound(gp)
    
    candidate, _ = optimize_acqf(UCB, bounds = torch.tensor([[args.min, args.max], [args.min, args.max]]), q=1, num_restarts=5)

    episodes = tqdm(range(args.n_episodes))
    print(candidate)

if __name__ == '__main__':
    main()
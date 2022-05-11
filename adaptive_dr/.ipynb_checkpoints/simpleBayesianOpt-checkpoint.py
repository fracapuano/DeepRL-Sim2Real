"""
Implementing Bayesian Optimisation
"""
from torch.distributions.uniform import Uniform
import argparse
import torch
import numpy as np

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
    parser.add_argument('--maxit', default=100, type=int, help='Maximal number of iterations for Bayesian Optimization')
    return parser.parse_args()
args = parse_args()
def main():
    
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
    phi_init = np.zeros((n_parameters, args.n_init))
    return_on_target = np.zeros(args.n_roll)

    """
    Populating each column of phi_init with lower (first) and upper (second) bound considered for the distribution
    sampled from the distribution having [min, max] as parameters
    """

    for init in range(args.n_init): 
        col_init = np.array([massesDistribution.sample().detach().numpy(), massesDistribution.sample().detach().numpy()]).reshape(-1,)
        phi_init[:, init] = col_init

        # testing the learned policy in the target environment for n_roll times
        roll_return = []
        for rollout in range(args.n_roll): 
            roll_return.append((col_init @ np.ones_like(col_init)) ** 2)
        
        roll_return = np.array(roll_return)
        print(roll_return.mean())
        return_on_target[init] = roll_return.mean()

    return_on_target = torch.tensor(return_on_target)
    return_on_target = standardize(return_on_target)

    # tensorifying Gaussian Process inputs
    return_on_target, phi_init = return_on_target, torch.tensor(phi_init)
    # fitting a Gaussian Process model
    gp = SingleTaskGP(return_on_target, phi_init)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    # building an acquisition function
    UCB = UpperConfidenceBound(gp)
    
    candidate, _ = optimize_acqf(UCB, bounds = torch.tensor([[args.min, args.max], [args.min, args.max]]), q=1, num_restarts=5)
    print(candidate)

if __name__ == '__main__':
    main()
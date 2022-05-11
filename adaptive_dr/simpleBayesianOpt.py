"""
Implementing Bayesian Optimisation
"""
from torch.distributions.uniform import Uniform
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer

import argparse
import torch
import numpy as np

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

def main():
    
    pbounds = {'x': (0, 8), 'y': (0, 8)}
    
    bounds_transformer = SequentialDomainReductionTransformer()

    mutating_optimizer = BayesianOptimization(
    f=blackbox_fun,
    pbounds=pbounds,
    verbose=1,
    random_state=1,
    bounds_transformer=bounds_transformer
    )

    mutating_optimizer.maximize(
    init_points=2,
    n_iter=50,
    )

    print(mutating_optimizer.max)

if __name__ == '__main__':
    main()
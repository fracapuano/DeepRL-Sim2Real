# from email import policy
from multiprocessing.managers import ValueProxy
from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.policy.init_weights()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        """
        This function updates the current parameters of the policy leveraging the 
        experience collected following that policy (i.e. with that specific set of 
        parameters)
        """
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        
        numberOfSteps = len(done)
        # initializing the sample of the log(pi(A|S, theta))
        policy_loss = []

        returns = np.zeros_like(rewards)
        
        for timestep in reversed(range(numberOfSteps)): 
            Gt = rewards[timestep] + (returns[timestep + 1] if timestep + 1 < numberOfSteps else 0)
            # Gt = (self.gamma ** torch.tensor(range(numberOfSteps - timestep))) @ rewards[timestep:]
            returns[timestep] = Gt
    
        # for timestep in reversed(range(numberOfSteps)): 
        #     Gt = rewards[timestep] + (returns[timestep + 1] if timestep + 1 < numberOfSteps else 0)
        #     value_function = self.baseline(states[timestep])
        #     returns[timestep] = Gt - value_function
        #     if done[timestep]:
        #         self.baseline_optimizer.zero_grad()
        #         value_function.backward()
        #         self.baseline_optimizer.step()


        returns = torch.tensor(returns)
        # implementing a baseline based on deviation by the mean
        returns = returns - returns.mean()

        # populating the sample of log(pi(A|S, theta))
        for log_prob, G in zip(action_log_probs, returns):
            policy_loss.append(log_prob * G)

        self.optimizer.zero_grad()

        # estimating log(pi(A|S, theta)) using MC procedure
        policy_loss = -1 * torch.stack(policy_loss).mean()

        # backpropagating
        policy_loss.backward()

        # stepping the optimizer
        self.optimizer.step()
        
        # reinitializing the data for the new upcoming training episode
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.states = []
        self.next_states = []
        del returns

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.policy.forward(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) =
            # log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2]))]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)            
# from email import policy
from multiprocessing.managers import ValueProxy
from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Agent(object):
    def __init__(self, policy, device='cpu', gamma=0.99, lr=1e-3, return_flag = "baseline"):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.policy.init_weights()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.return_flag = return_flag.lower()

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

        returns = torch.zeros_like(rewards)
        if self.return_flag == "standard": # considering all the rewards to obtain the trajectory return
            gammas = self.gamma ** torch.ones_like(rewards).cumsum(dim = 0)
            Gt = gammas @ rewards
            returns = returns + Gt 

        elif self.return_flag == "reward_to_go": # considering only the reward subsequent to having taken the action
            for timestep in reversed(range(numberOfSteps)): 
                Gt = rewards[timestep] + (self.gamma * returns[timestep + 1] if timestep + 1 < numberOfSteps else 0)
                returns[timestep] = Gt

        elif self.return_flag == "baseline": # reward to go introducing a baseline to standardize return
            for timestep in reversed(range(numberOfSteps)): 
                Gt = rewards[timestep] + (self.gamma * returns[timestep + 1] if timestep + 1 < numberOfSteps else 0)
                returns[timestep] = Gt
            returns = (returns - returns.mean())/returns.std()

        # populating the sample of log(pi(A|S, theta))
        for log_prob, G in zip(action_log_probs, returns):
            policy_loss.append(-1 * log_prob * G)

        self.optimizer.zero_grad()

        # estimating log(pi(A|S, theta)) using MC procedure
        policy_loss = torch.tensor(policy_loss, requires_grad=True).sum()

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

            # Compute Log probability of the action
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)            
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_value_estimate = torch.nn.Linear(self.hidden, 1)
    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value_estimate = self.fc3_value_estimate(x_critic)

        return normal_dist, value_estimate

class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.I = 1
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

        numberOfSteps = len(self.states)

        advantages = []
        gammas = torch.tensor([self.gamma**i for i in range(numberOfSteps)])
        G = []
        current_state_value = []
        next_state_value = []
        
        for idx in range(numberOfSteps):
            G.append(rewards[idx:] @ gammas[idx:])
            _, v_current = self.policy(states[idx])
            current_state_value.append(v_current)
            advantages.append(G[idx] - current_state_value[idx])
            _, v_next = self.policy(next_states[idx])
            next_state_value.append(v_next)

        G = torch.tensor(G, requires_grad=True)
        rewards = torch.tensor(rewards)

        advantages = torch.stack(advantages)
        current_state_value = torch.stack(current_state_value).squeeze(-1)
        next_state_value = torch.stack(next_state_value)

        bootstrap_state_value = self.gamma*next_state_value - current_state_value
        deltas = rewards + bootstrap_state_value

        actor_loss = torch.mean(-action_log_probs*advantages)
        loss_critic_fn = nn.MSELoss()
        critic_loss = loss_critic_fn(current_state_value, G)
            
        # for log_prob, advantage in zip(action_log_probs, advantages):
        #     actor_loss.append(- log_prob * self.I * advantage)

        # actor_loss = torch.tensor(actor_loss, requires_grad=True).mean()

        # for value_state, delta in zip(current_state_value, deltas):
        #     critic_loss.append(delta*value_state*self.I)

        # critic_loss = torch.tensor(critic_loss, requires_grad=True).mean()

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.I = self.gamma * self.I

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist, _ = self.policy.forward(x)

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
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

        actor_loss = -action_log_probs.item()*deltas.item()
        critic_loss = deltas.item()*current_state_value.item()

        total_loss = torch.tensor(actor_loss + critic_loss, requires_grad=True)
        
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


import gym
import argparse

from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--agent-type', default='reinforce', type=str, help='Select the agent that has to perform training reinforce - actorCritic - PPO - TRPO')
    parser.add_argument('--domain-type', default='source', type=str, help='source / target')
    return parser.parse_args()

args = parse_args()

def main():

    if args.domain_type.lower() == 'source':
        env = gym.make('CustomHopper-source-v0')
    elif args.domain_type.lower() == 'target':
        env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    if args.agent_type.lower() == 'reinforce':

        import agentReinforce
        policy = agentReinforce.Policy(observation_space_dim, action_space_dim)
        agent = agentReinforce.Agent(policy, device=args.device)
        actorCriticCheck=False

    elif args.agent_type.lower() == 'actorcritic':

        policy = Policy(observation_space_dim, action_space_dim)
        agent = Agent(policy)
        actorCriticCheck = True

    elif args.agent_type.lower() == 'ppo':

        actorCriticCheck=False
        from stable_baselines3 import PPO

        agent = PPO('MlpPolicy', env, verbose=1)
        agent.learn(total_timesteps=args.n_episodes)
        agent.save('ppo-model.mdl')
        return
    
    elif args.agent_type.lower() == 'trpo':

        actorCriticCheck=False
        from sb3_contrib.trpo.trpo import TRPO

        agent = TRPO('MlpPolicy', env, verbose=1)
        agent.learn(total_timesteps=args.n_episodes)
        agent.save('trpo-model.mdl')
        return

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:

            action, action_probabilities = agent.get_action(state)
            previous_state = state
            
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward
            
            agent.update_policy()
        
        if (episode+1)%args.print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)

    if actorCriticCheck:
        torch.save(agent.policy.state_dict(), "actorCritic-model.mdl")
    else:
        torch.save(agent.policy.state_dict(), "reinforce-model.mdl")

if __name__ == '__main__':
    main()
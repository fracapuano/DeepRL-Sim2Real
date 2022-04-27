import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

class Policy(torch.nn.Module):

    def __init__(self, state_space, action_space, hidden_layers, hidden_neurons, \
                activation_function, output_activation, init_sigma):
        """
        This constructor initializes a DNN with given parameters. 
        Parameters: 
            state_space: integer representing the cardinality of the state space
            action space: integer representing the cardinality of the action space
            hidden_layers: integer representing the number of hidden layers
            hidden_neurons: np.array of shape(hidden_layers,) in which the i-th element corresponds to the number of neurons
                            in the i-th layer
            activation_function: np.array of shape(hidden_layers - 1,) in which the i-th element corresponds to the i-th/i+1-th 
                                 activation function 
            output_activation: activation function to use on the output layer
            init_sigma: scalar used as variance for exploration of the action space

        Returns: 
            nn.Sequential() object
        """
        # init of the super class
        super().__init__()
        self.init_weights()

        # init for Policy
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function
        self.output_activation = output_activation
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        self.init_sigma = init_sigma
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        # ACTOR NETWORK
        # state-space to first hidden layer
        layers = [nn.Linear(self.state_space, self.hidden_neurons[0]), self.activation_function[0]()]

        # first layer to last hidden layer
        for j in range(self.hidden_layers-1):
            act = self.activation_function[j]
            layers += [nn.Linear(self.hidden_neurons[j], self.hidden_neurons[j+1]), act()]
        
        # last hidden layer to output layer
        layers += [nn.Linear(self.hidden_neurons[-1], self.action_space), self.output_activation]
        
        ### CRITIC PART OF THE NETWORK FROM HERE ON ###
        
        return nn.Sequential(*layers)

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
        # TODO 2.2.b: forward in the critic network

        
        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

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

        #
        # TODO 2.2.a:
        #             - compute discounted returns
        #             - compute policy gradient loss function given actions and returns
        #             - compute gradients and step the optimizer
        #


        #
        # TODO 2.2.b:
        #             - compute boostrapped discounted return estimates
        #             - compute advantage terms
        #             - compute actor loss and critic loss
        #             - compute gradients and step the optimizer
        #

        return        

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


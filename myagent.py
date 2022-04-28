from multiprocessing.managers import ValueProxy
from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import Rewards

class Policy(torch.nn.Module):

    def __init__(self, state_space, action_space,
                hidden_layers = 3, hidden_neurons = np.array([64, 64, 32]),
                activation_function = np.array([torch.nn.ReLU for _ in range(3)]),
                output_activation = torch.nn.Identity, init_sigma = 0.5):
        """
        This constructor initializes a DNN with given parameters. 
        Parameters: 
            state_space: integer representing the cardinality of the state space
            action space: integer representing the cardinality of the action space
            hidden_layers: integer representing the number of hidden layers
            hidden_neurons: np.array of shape(hidden_layers,) in which the i-th element corresponds to the number of neurons
                            in the i-th layer
            activation_function: np.array of shape(hidden_layers,) in which the i-th element corresponds to the i-th/i+1-th 
                                 activation function 
            output_activation: activation function to use on the output layer
            init_sigma: scalar used as variance for exploration of the action space

        Returns: 
            None
        """
        # init of the super class
        super().__init__()
        
        # init for Policy
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation_function = activation_function
        self.output_activation = output_activation

        self.ActorNetwork = None
        self.weight_initialized = False

        # number of hidden layers must be consistent with hidden neurons array 
        if len(self.hidden_neurons) != self.hidden_layers: 
            raise ValueError("The number of layers is inconsistent with the hidden neurons array!")
        
        # number of activation function must be consistent with number of hidden layers
        # if len(self.activation_function) != self.hidden_layers: 
        #     raise ValueError("The number of activation functions is inconsistent with the number of hidden layers!")

        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        self.init_sigma = init_sigma
        self.sigma = nn.Parameter(torch.zeros(self.action_space)+init_sigma)

    def actor_network(self): 
        """
        This function builds the neural network with respect to the initializations on the parameters previously
        declared. 
        Paramethers: 
            None
        Returns: 
            nn.Sequential() object
        """
        # state-space to first hidden layer
        layers = [nn.Linear(int(self.state_space), int(self.hidden_neurons[0])), self.activation_function[0]()]

        # first layer to last hidden layer
        for j in range(1, self.hidden_layers-1):
            act = self.activation_function[j]
            layers += [nn.Linear(int(self.hidden_neurons[j]), int(self.hidden_neurons[j+1])), act()]
        
        # last hidden layer to output layer
        layers += [nn.Linear(int(self.hidden_neurons[-1]), self.action_space), self.output_activation()]
        
        # imposing the network as class attribute
        self.ActorNetwork = nn.Sequential(*layers)
        
        ### CRITIC PART OF THE NETWORK FROM HERE ON ###
        
    def init_weights(self):
        """
        This function initializes the weights and the bias per each neuron in each Linear hidden layer
        """
        self.actor_network()

        for m in self.ActorNetwork.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
                  
        self.weight_initialized = True

    def forward(self, x):
        """
        This function passes an input vector x into the network, adds some random noise to grant exploration and outputs
        a vector. 
        Parameters: 
            x: np.array of shape (self.state_space, )
        Returns: 
            out: np.array of shape (self.action_space, )
        """
        # checking whether or not the weights/bias have been initialized
        if not self.weight_initialized: 
            raise ValueError("The weights have not been initialized yet! Call init_weight() prior to forward!")
        
        # casting numpy array to tensor
        # x = torch.from_numpy(x).float()

        for layer in self.ActorNetwork:
            x = layer(x)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(x, sigma)

        # the output of this network is a probability distribution
        return normal_dist

class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.ActorNetwork.parameters(), lr=1e-3)

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
        rew = Rewards()

        numberOfEpisodes = len(done)
        for episode in range(numberOfEpisodes): 
            G = rew.discount_rewards(rewards[episode:], self.gamma)
            print("**** INSIDE EACH EPISODE ****")
            print(G)
            
            pseudo_loss = torch.sum(action_log_probs * G)
            print(pseudo_loss)
            
            self.optimizer.zero_grad()
            pseudo_loss.backward()
            self.optimizer.step()
        
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
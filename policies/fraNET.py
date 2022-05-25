import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCriticPolicy(torch.nn.Module):

    def __init__(self, state_space, action_space,
                hidden_layers = 2, hidden_neurons = np.array([64 for _ in range(2)]),
                activation_function = np.array([torch.nn.Tanh for _ in range(2)]),
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
        self.CriticNetwork = None
        self.weight_initialized = False

        # number of hidden layers must be consistent with hidden neurons array 
        if len(self.hidden_neurons) != self.hidden_layers: 
            raise ValueError("The number of layers is inconsistent with the hidden neurons array!")

        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        self.init_sigma = init_sigma
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

    def actor_network(self): 
        """
        This function builds the actor neural network with respect to the initializations on the parameters previously
        declared. 
        Paramethers: 
            None
        Returns: 
            nn.Sequential() object
        """
        # state-space to first hidden layer
        layers = [torch.nn.Linear(int(self.state_space), int(self.hidden_neurons[0])), self.activation_function[0]()]

        # first layer to last hidden layer
        for j in range(1, self.hidden_layers-1):
            act = self.activation_function[j]
            layers += [torch.nn.Linear(int(self.hidden_neurons[j]), int(self.hidden_neurons[j+1])), act()]
        
        # last hidden layer to output layer
        layers += [torch.nn.Linear(int(self.hidden_neurons[-1]), self.action_space), self.output_activation()]
        
        # imposing the network as class attribute
        self.ActorNetwork = torch.nn.Sequential(*layers)
    
    def critic_network(self): 
        """
        This function builds the critic neural network with respect to the initializations on the parameters previously
        declared. 
        Paramethers: 
            None
        Returns: 
            nn.Sequential() object
        """
        # state-space to first hidden layer
        layers = [torch.nn.Linear(int(self.state_space), int(self.hidden_neurons[0])), self.activation_function[0]()]

        # first layer to last hidden layer
        for j in range(1, self.hidden_layers-1):
            act = self.activation_function[j]
            layers += [torch.nn.Linear(int(self.hidden_neurons[j]), int(self.hidden_neurons[j+1])), act()]
        
        # last hidden layer to output layer
        layers += [torch.nn.Linear(int(self.hidden_neurons[-1]), 1), self.output_activation()]
        
        # imposing the network as class attribute
        self.CriticNetwork = torch.nn.Sequential(*layers)
        
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
    
        # instantiating the network when the network is not instantiated
        if self.ActorNetwork is None: 
            self.actor_network()
        if self.CriticNetwork is None: 
            self.critic_network()
 
        x_actor, x_critic = self.ActorNetwork(x), self.CriticNetwork(x)
        
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(x_actor, sigma)

        # the output of this network is a probability distribution
        return normal_dist, x_critic

class ReinforcePolicy(torch.nn.Module):

    def __init__(self, state_space, action_space,
                hidden_layers = 3, hidden_neurons = np.array([64, 64, 32]),
                activation_function = np.array([torch.nn.Tanh for _ in range(3)]),
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

        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        self.init_sigma = init_sigma
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

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
        layers = [torch.nn.Linear(int(self.state_space), int(self.hidden_neurons[0])), self.activation_function[0]()]

        # first layer to last hidden layer
        for j in range(1, self.hidden_layers-1):
            act = self.activation_function[j]
            layers += [torch.nn.Linear(int(self.hidden_neurons[j]), int(self.hidden_neurons[j+1])), act()]
        
        # last hidden layer to output layer
        layers += [torch.nn.Linear(int(self.hidden_neurons[-1]), self.action_space), self.output_activation()]
        
        # imposing the network as class attribute
        self.ActorNetwork = torch.nn.Sequential(*layers)
        
        
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
        if self.ActorNetwork is None: 
            # instantiating the network when the network is not instantiated
            self.actor_network()

        # initializing the weigths
        if not self.weight_initialized: 
            self.init_weights()
 
        for layer in self.ActorNetwork:
            x = layer(x)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(x, sigma)

        # the output of this network is a probability distribution
        return normal_dist
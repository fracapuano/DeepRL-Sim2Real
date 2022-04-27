from agent import *
from pprint import pprint

layers = 100

params = {
    "state_space": 11, 
    "action_space": 3, 
    "hidden_layers": layers, 
    "hidden_neurons": 32 * np.ones(layers, dtype = np.int16), 
    "activation_function": np.array([nn.ReLU for _ in range(layers)]), 
    "output_activation": nn.Softmax,
    "init_sigma": 0.5
}

policy = Policy(**params)

print(policy.actor_network())
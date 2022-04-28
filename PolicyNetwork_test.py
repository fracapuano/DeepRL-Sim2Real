from numpy import dtype
from myagent import *

layers = 3

params = {
    "state_space": 11, 
    "action_space": 3, 
    "hidden_layers": layers, 
    "hidden_neurons": 5 * np.ones(layers, dtype = np.int16), 
    "activation_function": [nn.ReLU for _ in range(layers)], 
    "output_activation": nn.Softmax,
    "init_sigma": 0.5
}

policy = Policy(**params)
policy.actor_network()
print(policy.ActorNetwork)

# testing the forward method
random_input = np.random.random(params["state_space"]).astype(np.double)

policy.init_weights()

print(policy.forward(random_input))
import torch
from torch import nn
from torch.nn import functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers = [64, 64]):
        """ Build a fully-connected network with arbitrary hidden layers.
        Sub-classing from 'nn.Module' tracks the attributes added to this class
        & randomly initializes weights & biases for each of the defined layer.
        
        Params:
            state_size (int)    ---  dimension of each state (input to model)
            action_size (int)   ---  dimension of each action (output)
            seed (int)          ---  random seed
            hidden_layers (int) ---  no. of nodes in hidden layers """
        
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)
        

    def forward(self, state):
        """ Build a network that maps state -> action values """
        
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
        
        return self.output(state)
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def hidden_init(layer):
    """Initialization scheme for the weights of a layer.
     Params
        ======
           layer: the layer you want to initialize
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)




class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, size_1=256, size_2=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            size_1 (int): Number of nodes in first hidden layer
            size_2 (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        self.action_size=action_size
        self.dense_1 = nn.Linear(state_size, size_1)
        self.dense_2 = nn.Linear(size_1, size_2)
        self.dense_3 = nn.Linear(size_2, self.action_size)
        self.reset_parameters()
        
    
        
    def reset_parameters(self):
        """Initializing the weights of the model"""
        self.dense_1.weight.data.uniform_(*hidden_init(self.dense_1))
        self.dense_2.weight.data.uniform_(*hidden_init(self.dense_2))
        self.dense_3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions.
        Params
        ======
            state (array_like): current state
        """
        x=self.dense_1(state)
        x=F.relu(x)
        x=self.dense_2(x)
        x=F.relu(x)
        x=self.dense_3(x)     
        
        return F.tanh(x)

    
class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size,action_size, seed, size_1=256, size_2=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            size_1 (int): Number of nodes in the first hidden layer
            size_2 (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.dense_1 = nn.Linear(state_size, size_1)
        self.dense_2 = nn.Linear(size_1+action_size, size_2)
        self.dense_3 = nn.Linear(size_2, 1)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        """Initializing the weights of the model"""
        self.dense_1.weight.data.uniform_(*hidden_init(self.dense_1))
        self.dense_2.weight.data.uniform_(*hidden_init(self.dense_2))
        self.dense_3.weight.data.uniform_(-3e-3, 3e-3)
        


    def forward(self, state,action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.
        Params
        ======
            state (array_like): current state
            action (array_like): current action
        """
        x=self.dense_1(state)
        x=F.relu(x)        
        x=torch.cat((x,action), dim=1)
        x=self.dense_2(x)
        x=F.relu(x)        
        
        return self.dense_3(x)






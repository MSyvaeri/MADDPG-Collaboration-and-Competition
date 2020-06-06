import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np




def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, size_1=256, size_2=256):
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size=action_size
        self.dense_1 = nn.Linear(state_size, size_1)
        self.dense_2 = nn.Linear(size_1, size_2)
        self.dense_3 = nn.Linear(size_2, self.action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        
        self.dense_1.weight.data.uniform_(*hidden_init(self.dense_1))
        self.dense_2.weight.data.uniform_(*hidden_init(self.dense_2))
        self.dense_3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state):
        
        x=self.dense_1(state)
        x=F.relu(x)
        x=self.dense_2(x)
        x=F.relu(x)
        x=self.dense_3(x)     
        
        return F.tanh(x)

    
class Critic(nn.Module):

    def __init__(self, state_size,action_size, seed, size_1=256, size_2=256):
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dense_1 = nn.Linear(state_size, size_1)
        self.dense_2 = nn.Linear(size_1+action_size, size_2)
        self.dense_3 = nn.Linear(size_2, 1)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.dense_1.weight.data.uniform_(*hidden_init(self.dense_1))
        self.dense_2.weight.data.uniform_(*hidden_init(self.dense_2))
        self.dense_3.weight.data.uniform_(-3e-3, 3e-3)
        


    def forward(self, state,action):
        
        x=self.dense_1(state)
        x=F.relu(x)        
        x=torch.cat((x,action), dim=1)
        x=self.dense_2(x)
        x=F.relu(x)        
        
        return self.dense_3(x)






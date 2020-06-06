#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
from collections import namedtuple, deque

import copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2             # for soft update of target parameters
LR_ACTOR = 1e-4               # learning rate
LR_CRITIC = 1e-3               # learning rate 
UPDATE_EVERY = 2     # how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, n_agents,seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents=n_agents
        self.seed = random.seed(seed)
        

        # Actor-Network
        self.actor_network_local = Actor(state_size, action_size, seed).to(device)
        self.actor_network_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network_local.parameters(), lr=LR_ACTOR)
        
        # Actor-Network
        self.critic_network_local = Critic(state_size, action_size, seed).to(device)
        self.critic_network_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_network_local.parameters(), lr=LR_CRITIC)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        
        ## Introduce Noise
        self.hard_update(self.actor_network_target, self.actor_network_local)
        self.hard_update(self.critic_network_target,self.critic_network_local)
        
    def hard_update(self,target,source):
        for target_params,source_params in zip(target.parameters(),source.parameters()):
            target_params.data.copy_(source_params.data)
        
        
    def step(self, states, actions, rewards, next_states, dones,episode, learn=True):
        # Save experience in replay memory
        for i in range(self.n_agents):
            self.memory.add(states[i],actions[i],rewards[i],next_states[i],dones[i])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE and learn:
                if episode >-1:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                    
    def act(self, state, episode,eps):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if episode<-1:
            action=np.random.uniform(-1,1,size=(np.shape(state)[0],self.action_size))
        else:
            state = torch.from_numpy(state).float().to(device)
            self.actor_network_local.eval()
            with torch.no_grad():
                action_values = self.actor_network_local(state).cpu().data.numpy()
            self.actor_network_local.train()
            action=action_values
            action=action+eps*np.random.normal(0,1,size=np.shape(action))
         #   print(self.noise.sample())
        return np.clip(action,-1,1)
        
        
    def learn(self, experiences, gamma):
       
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        next_actions=self.actor_network_target(next_states)
        
        Q_targets_next = self.critic_network_target(next_states,next_actions)
        # Compute Q targets for current states 
        
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.critic_network_local(states,actions)
        # Compute loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network_local.parameters(), 1)
        self.critic_optimizer.step()

   
        
        # ------------------- train actor network ------------------- #
        action_pred=self.actor_network_local(states)
        actor_loss=-self.critic_network_local(states,action_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        

        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_network_local, self.actor_network_target, TAU)    
        self.soft_update(self.critic_network_local, self.critic_network_target, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
       
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    


    
            
            

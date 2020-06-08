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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, n_agents,seed,
                 buffer_size=int(1e5),batch_size=128,gamma=0.99,tau=1e-2,
                lr_actor=1e-4,lr_critic=1e-3,update_every=2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int): Number of agents which are parallel used 
            seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): factor for soft update of target parameters
            lr_actor (float): learning rate of the actor
            lr_critic (float): learning rate of the critic
            update_every (int): number of steps between the updates of the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents=n_agents
        self.gamma=gamma
        self.tau=tau
        self.update_every=update_every
        self.batch_size=batch_size
        np.random.seed(seed)
        

        # Actor-Network
        self.actor_network_local = Actor(state_size, action_size, seed).to(device)
        self.actor_network_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network_local.parameters(), lr=lr_actor)
        
        # Actor-Network
        self.critic_network_local = Critic(state_size, action_size, seed).to(device)
        self.critic_network_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_network_local.parameters(), lr=lr_critic)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0
        
        
        ##Copy local and target network into each other so that they have the same starting point
        self.hard_update(self.actor_network_target, self.actor_network_local)
        self.hard_update(self.critic_network_target,self.critic_network_local)
        
    def hard_update(self,target,source):
        """initialize local and target network with the same weights
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        ##initialize local and target network with the same weights
        for target_params,source_params in zip(target.parameters(),source.parameters()):
            target_params.data.copy_(source_params.data)
        
        
    def step(self, states, actions, rewards, next_states, dones, learn=True):
        """Save experience in replay memory, and use random sample from buffer to learn.
        Params
        ======
            state (array_like): current states
            actions (array_like): current actions
            rewards (array_like): current rewards
            next_states (array_like): current states
            dones (boolian): boolian whether this episode is finished
            learn (boolian): boolian whether the networks should be trained
        
        """
        # Save experience in replay memory
        for i in range(self.n_agents):
            self.memory.add(states[i],actions[i],rewards[i],next_states[i],dones[i])
        
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size and learn:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state,eps):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon which regularizes the size of the added noise
        """
        
        state = torch.from_numpy(state).float().to(device)
        self.actor_network_local.eval()
        with torch.no_grad():
            action_values = self.actor_network_local(state).cpu().data.numpy()
        self.actor_network_local.train()
        
        ## I tested OUNoise as well, but the results were better just with random noise from a normal distribution
        
        action=action_values+eps*np.random.normal(0,1,size=np.shape(action_values))   
        return np.clip(action,-1,1)
    
        
        
    def learn(self, experiences):      
        """Summary function for the update process for the critic and the actor, softupdate for the target networks
        
        Params
        ======
            experiences (tuple): tuple of states, actions, rewards, next_states, dones
        """
        states, actions, rewards, next_states, dones = experiences
        
        self.update_critic(states,actions,rewards,next_states,dones)
        self.update_actor(states)
        self.update_target_networks()
        
        
    def update_critic(self,states,actions,rewards,next_states,dones):
        """Update process for the critic
        
        Params
        ======
            state (array_like): current states
            actions (array_like): current actions
            rewards (array_like): current rewards
            next_states (array_like): current states
            dones (boolian): boolian whether this episode is finished
        """
        
        # Get max predicted Q values (for next states) from target model
        next_actions=self.actor_network_target(next_states)
        
        Q_targets_next = self.critic_network_target(next_states,next_actions)
        # Compute Q targets for current states 
        
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.critic_network_local(states,actions)
        # Compute loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network_local.parameters(), 1)
        self.critic_optimizer.step()

   
        
    def update_actor(self,states):
        """Update process for the actor
        
        Params
        ======
            state (array_like): current states
        """
                     
        action_pred=self.actor_network_local(states)
        actor_loss=-self.critic_network_local(states,action_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        

    def update_target_networks(self):
        """Soft update for the target networks"""
        
        self.soft_update(self.actor_network_local, self.actor_network_target)    
        self.soft_update(self.critic_network_local, self.critic_network_target)                     


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
 
            
            

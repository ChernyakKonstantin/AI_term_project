import random
import numpy as np
import torch
import torch.nn

class Agent:
    def __init__(self, environment, expirience_buffer, device):
        self.env = environment
        self.exp_buffer = expirience_buffer
        self.device = device
        
        self.total_reward = 0.0
        self.state = None
        
        self._reset()
        

    def _reset(self):
        self.state = self.env.reset()            
        self.total_reward = 0.0
    
    
    def train_step(self, net, epsilon):
        """
        The method perform a step in environment.

        Parameters
        ----------
        net : instance of class Net.
        epsilon : float

        Returns
        -------
        done_reward : float
            Reward, obrained through the episode.

        """
        action = self._select_action(net, epsilon)

        next_state, reward, is_done = self.env.step(action)

        self.total_reward += reward
        
        self.exp_buffer.append(self.state, next_state, action, reward, is_done)
        
        self.state = next_state
        
        if is_done == True:
            done_reward = self.total_reward
            self._reset()
            return done_reward
        else:
            return None
            
        
    def _select_action(self, net, epsilon):
        """
        The method selects random action with probability of epsilon.
        Otherwise it selects action, maximazing Q value.
        Self.state is preprocessed before convertation to torch.tensor.

        Parameters
        ----------
        net : instance of class Net.
        epsilon : float

        Returns
        -------
        action : int
        """
        if random.random() < epsilon:
            action = self.env.sample_action()
        else:
            state_v_l = torch.unsqueeze(torch.FloatTensor(self.state[0]/255), dim=0).to(self.device) #torch.tensor[batch_size, n_channels, width, height]
            state_v_r = torch.unsqueeze(torch.FloatTensor(self.state[1]/255), dim=0).to(self.device) #torch.tensor[batch_size, n_channels, width, height]
            q_vals_v = net(state_v_l, state_v_r)
            action = torch.argmax(q_vals_v, dim=1)[0].item()
        return action
        
    
    def eval_step(self, net):
        state_v_l = torch.unsqueeze(torch.FloatTensor(self.state[0]/255), dim=0).to(self.device) #torch.tensor[batch_size, n_channels, width, height]
        state_v_r = torch.unsqueeze(torch.FloatTensor(self.state[1]/255), dim=0).to(self.device) #torch.tensor[batch_size, n_channels, width, height]
        q_vals_v = net(state_v_l, state_v_r)
        action = torch.argmax(q_vals_v, dim=1)[0].item()

        next_state, reward, is_done = self.env.step(action)
        self.state = next_state
        self.total_reward += reward

        if is_done == True:
            done_reward = self.total_reward
            self._reset()
            return done_reward
        else:
            return None    
        
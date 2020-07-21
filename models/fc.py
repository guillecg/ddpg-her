import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)

    return -lim, lim


class ActorFC(nn.Module):
    """Initialize parameters and build model.

    Keywords:
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        seed (int): Random seed

    Return:
        action output of network with tanh activation

    """

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed=42):
        super(ActorFC, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        action = torch.tanh(self.fc3(h))

        return action


class CriticFC(nn.Module):
    """Initialize parameters and build model.

    Keywords:
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        seed (int): Random seed

    Return:
        value output of network

    """

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed=42):
        super(CriticFC, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Q1 architecture
        self.critic_1_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.critic_1_fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic_1_fc3 = nn.Linear(fc2_units, 1)

        # Q2 architecture
        self.critic_2_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.critic_2_fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic_2_fc3 = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.critic_1_fc1.weight.data.uniform_(*hidden_init(self.critic_1_fc1))
        self.critic_1_fc2.weight.data.uniform_(*hidden_init(self.critic_1_fc2))
        self.critic_1_fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.critic_2_fc1.weight.data.uniform_(*hidden_init(self.critic_2_fc1))
        self.critic_2_fc2.weight.data.uniform_(*hidden_init(self.critic_2_fc2))
        self.critic_2_fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)

        h1 = F.relu(self.critic_1_fc1(state_action))
        h1 = F.relu(self.critic_1_fc2(h1))
        critic_1_q_value = self.critic_1_fc3(h1)

        h2 = F.relu(self.critic_2_fc1(state_action))
        h2 = F.relu(self.critic_2_fc2(h2))
        critic_2_q_value = self.critic_2_fc3(h2)

        return critic_1_q_value, critic_2_q_value

    def Q1(self, state, action):
        state_action = torch.cat([state, action], dim=1)

        h = F.relu(self.critic_1_fc1(state_action))
        h = F.relu(self.critic_1_fc2(h))
        q_value = self.critic_1_fc3(h)

        return q_value

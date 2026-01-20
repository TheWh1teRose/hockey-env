import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


    def forward(self, state):
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        mu = self.fc3(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        u = mu + torch.randn_like(mu) * std
        action = torch.tanh(u)
        return action
    
    def sample_with_logprob(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        u = mu + torch.randn_like(mu) * std
        log_prob = -0.5 * ((u - mu) / (std + 1e-6)).pow(2) - log_std - 0.5 * np.log(2 * np.pi) - torch.log(1 - torch.tanh(u).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = torch.tanh(u)
        return action, log_prob

    def act(self, state):
        # deterministic action
        mu, _ = self.forward(state)
        return torch.tanh(mu)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        h = F.relu(self.fc1(torch.cat([state, action], dim=-1)))
        h = F.relu(self.fc2(h))
        return self.fc3(h)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .models import Actor, Critic
import copy
import torch.optim as optim
import torch
from .replay_buffer import PrioritizedReplayBuffer

class SAC:
    def __init__(self, buffer, state_dim, action_dim, hidden_dim, lr, gamma, tau, alpha, device):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)
        self.alpha = alpha
        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(alpha), device=device))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.gamma = gamma
        self.tau = tau

        self.is_prioritized_buffer = isinstance(buffer, PrioritizedReplayBuffer)
        
        self.device = device
        self.target_entropy = torch.tensor(-action_dim, device=device)
        self.batch_size = 100
        self.buffer = buffer
    
    def update(self):
        if self.is_prioritized_buffer:
            state, action, reward, next_state, done, weights, indices = self.buffer.sample(self.batch_size)
        else:
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample_with_logprob(next_state)
            q1_t = self.target_critic_1(next_state, next_action)
            q2_t = self.target_critic_2(next_state, next_action)
            min_q_t = torch.min(q1_t, q2_t)
            alpha = self.log_alpha.exp()
            next_q_t = min_q_t - alpha * next_log_prob
            target_q_t = reward + (1 - done) * self.gamma * next_q_t
        q1_val = self.critic_1(state, action)
        q2_val = self.critic_2(state, action)

        if self.is_prioritized_buffer:
            critic_loss = F.mse_loss(q1_val, target_q_t, weight=weights) + F.mse_loss(q2_val, target_q_t, weight=weights)
            td_error = torch.abs(target_q_t - q1_val) + torch.abs(target_q_t - q2_val)
            self.buffer.update_priorities(indices, td_error.detach().cpu().numpy())
        else:
            critic_loss = F.mse_loss(q1_val, target_q_t) + F.mse_loss(q2_val, target_q_t)
            

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Actor update
        a_pi, logp_pi = self.actor.sample_with_logprob(state)
        q1_pi = self.critic_1(state, a_pi)
        q2_pi = self.critic_2(state, a_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * logp_pi - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Return metrics dict for logging
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            "q1_mean": q1_val.mean().item(),
            "q2_mean": q2_val.mean().item(),
            "log_prob_mean": logp_pi.mean().item(),
        }

    def act(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(self.device)
            else:
                state = state.to(self.device)
            action, _ = self.actor.sample_with_logprob(state)
            return action.cpu().numpy()

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'log_alpha': self.log_alpha.data,
            'alpha': self.alpha
        }, path)
    
    def load(self, path):
        data = torch.load(path)
        self.actor.load_state_dict(data['actor'])
        self.critic_1.load_state_dict(data['critic_1'])
        self.critic_2.load_state_dict(data['critic_2'])
        self.target_critic_1.load_state_dict(data['target_critic_1'])
        self.target_critic_2.load_state_dict(data['target_critic_2'])
        self.log_alpha.data = data['log_alpha']
        self.alpha = data['alpha']
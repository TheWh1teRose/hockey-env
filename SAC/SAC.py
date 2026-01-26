import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from .models import Actor, Critic
import copy
from .replay_buffer import PrioritizedReplayBuffer

class SAC:
    def __init__(self, buffer, state_dim, action_dim, hidden_dim, lr, gamma, tau, alpha, device, 
                 max_grad_norm=1.0, batch_size=512, use_amp=True, use_compile=True, updates_per_step=4):
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.updates_per_step = updates_per_step
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)
        
        # Apply torch.compile() for JIT optimization (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            print("Applying torch.compile() to networks...")
            self.actor = torch.compile(self.actor, mode="reduce-overhead")
            self.critic_1 = torch.compile(self.critic_1, mode="reduce-overhead")
            self.critic_2 = torch.compile(self.critic_2, mode="reduce-overhead")
            self.target_critic_1 = torch.compile(self.target_critic_1, mode="reduce-overhead")
            self.target_critic_2 = torch.compile(self.target_critic_2, mode="reduce-overhead")
        
        self.alpha = alpha
        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(alpha), device=device))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        self.is_prioritized_buffer = isinstance(buffer, PrioritizedReplayBuffer)
        
        self.target_entropy = torch.tensor(-action_dim, device=device, dtype=torch.float32)
        self.batch_size = batch_size
        self.buffer = buffer
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)
        
        if self.use_amp:
            print(f"Mixed precision training (AMP) enabled")
        print(f"Batch size: {self.batch_size}, Updates per step: {self.updates_per_step}")
    
    def update(self):
        # Accumulators for metrics
        metrics_acc = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "alpha_loss": 0.0,
            "alpha": 0.0,
            "q1_mean": 0.0,
            "q2_mean": 0.0,
            "log_prob_mean": 0.0,
        }
        
        for _ in range(self.updates_per_step):
            metrics = self._single_update()
            for key in metrics_acc:
                metrics_acc[key] += metrics[key]
        
        # Average metrics
        for key in metrics_acc:
            metrics_acc[key] /= self.updates_per_step
        
        return metrics_acc
    
    def _single_update(self):
        """Perform a single gradient update step with AMP."""
        if self.is_prioritized_buffer:
            state, action, reward, next_state, done, weights, indices = self.buffer.sample(self.batch_size)
        else:
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        # Critic update with mixed precision
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                next_action, next_log_prob = self.actor.sample_with_logprob(next_state)
                q1_t = self.target_critic_1(next_state, next_action)
                q2_t = self.target_critic_2(next_state, next_action)
                min_q_t = torch.min(q1_t, q2_t)
                alpha = self.log_alpha.exp()
                next_q_t = min_q_t - alpha * next_log_prob
                target_q_t = reward + (1 - done) * self.gamma * next_q_t
        
        with autocast(enabled=self.use_amp):
            q1_val = self.critic_1(state, action)
            q2_val = self.critic_2(state, action)

            if self.is_prioritized_buffer:
                # Weighted MSE for prioritized replay
                td1 = (q1_val - target_q_t).pow(2)
                td2 = (q2_val - target_q_t).pow(2)
                critic_loss = (weights * td1).mean() + (weights * td2).mean()
            else:
                critic_loss = F.mse_loss(q1_val, target_q_t) + F.mse_loss(q2_val, target_q_t)

        # Update priorities (outside autocast to avoid precision issues)
        if self.is_prioritized_buffer:
            with torch.no_grad():
                td_error = torch.abs(target_q_t - q1_val) + torch.abs(target_q_t - q2_val)
                self.buffer.update_priorities(indices, td_error.cpu().numpy())

        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_1_optimizer)
        self.scaler.unscale_(self.critic_2_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.scaler.step(self.critic_1_optimizer)
        self.scaler.step(self.critic_2_optimizer)

        # Actor update with mixed precision
        with autocast(enabled=self.use_amp):
            a_pi, logp_pi = self.actor.sample_with_logprob(state)
            q1_pi = self.critic_1(state, a_pi)
            q2_pi = self.critic_2(state, a_pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha * logp_pi - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.scaler.step(self.actor_optimizer)

        # Alpha update (keep in fp32 for stability)
        alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Update scaler
        self.scaler.update()

        # Soft update target networks
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
                state = torch.from_numpy(state).float().to(self.device, non_blocking=True)
            else:
                state = state.to(self.device, non_blocking=True)
            with autocast(enabled=self.use_amp):
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
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(data['actor'])
        self.critic_1.load_state_dict(data['critic_1'])
        self.critic_2.load_state_dict(data['critic_2'])
        self.target_critic_1.load_state_dict(data['target_critic_1'])
        self.target_critic_2.load_state_dict(data['target_critic_2'])
        self.log_alpha.data = data['log_alpha']
        self.alpha = data['alpha']
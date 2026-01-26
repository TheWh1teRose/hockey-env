"""

Class that represents the self-play opponent.
"""

from ..opponent import Opponent
import torch
import numpy as np
from SAC.helpers import normalize_obs
from SAC.models import Actor

class SelfPlayOpponent(Opponent):
    def __init__(self, name: str, model_path: str = None, Actor: Actor = None, device: str = "cpu"):
        if model_path is not None:
            self.model = self._load_actor(model_path, device)
            self.model.eval()
        elif Actor is not None:
            self.model = Actor.to(device)
            self.model.eval()
        self.device = device
        self.name = name

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Get actions for observations.
        
        Supports both single observations (obs_dim,) and batched observations (batch_size, obs_dim).
        
        Args:
            obs: Observation(s) with shape (obs_dim,) or (batch_size, obs_dim)
            
        Returns:
            Actions with shape (action_dim,) or (batch_size, action_dim)
        """
        with torch.no_grad():
            obs = normalize_obs(obs)
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            
            # Handle both single and batched observations
            is_single = obs_tensor.dim() == 1
            if is_single:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            actions = self.model.act(obs_tensor)
            
            if is_single:
                actions = actions.squeeze(0)
            
            # Convert GPU tensor to numpy for environment
            return actions.cpu().numpy()

    def _load_actor(self,checkpoint_path: str, device: str) -> Actor:
        """Load only the actor network from a checkpoint."""
        state_dim = 18
        action_dim = 4
        hidden_dim = 128
        actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()
        return actor

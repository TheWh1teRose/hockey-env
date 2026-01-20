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
        with torch.no_grad():
            obs = normalize_obs(obs)
            obs = torch.from_numpy(obs).float().to(self.device)
            return self.model.act(obs)

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
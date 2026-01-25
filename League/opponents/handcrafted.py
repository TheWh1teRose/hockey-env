from League.opponent import Opponent
import hockey.hockey_env as h_env
import numpy as np

class HandcraftedOpponent(Opponent):
    def __init__(self, name: str, setup_name: str):
        self.name = name
        self.opponent = None
        if setup_name == "strong":
            self.opponent = h_env.BasicOpponent(weak=False)
        elif setup_name == "weak":
            self.opponent = h_env.BasicOpponent(weak=True)
        else:
            raise ValueError(f"Unknown setup name: {setup_name}")

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Get actions for observations.
        
        Supports both single observations (obs_dim,) and batched observations (batch_size, obs_dim).
        
        Args:
            obs: Observation(s) with shape (obs_dim,) or (batch_size, obs_dim)
            
        Returns:
            Actions with shape (action_dim,) or (batch_size, action_dim)
        """
        # Handle both single and batched observations
        if obs.ndim == 1:
            # Single observation
            return self.opponent.act(obs)
        else:
            # Batched observations - process each one
            actions = np.array([self.opponent.act(o) for o in obs])
            return actions

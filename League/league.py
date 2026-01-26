"""
Class that manages the league of agents.
"""

import os
import numpy as np
from .opponent import Opponent
from .opponents.self import SelfPlayOpponent
from .opponents.handcrafted import HandcraftedOpponent
from play import play
import random

class League:
    def __init__(self, config: dict):
        self.config = config
        self.opponents = []
        self.win_rates = {}
        
        self.current_opponent = None

        init_opponents = []
        for opponent_config in config["init_opponents"]:
            if opponent_config["type"] == "checkpoint":
                init_opponents.append(SelfPlayOpponent(opponent_config["name"], opponent_config["model_path"], config["training"]["device"]))
            elif opponent_config["type"] == "handcrafted":
                init_opponents.append(HandcraftedOpponent(opponent_config["name"], opponent_config["strength"]))
        
        for opponent in init_opponents:
            self.add_opponent(opponent)
        
        self.new_opponent()


    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Get actions for observations.
        
        Supports both single observations (obs_dim,) and batched observations (batch_size, obs_dim).
        
        Args:
            obs: Observation(s) with shape (obs_dim,) or (batch_size, obs_dim)
            
        Returns:
            Actions with shape (action_dim,) or (batch_size, action_dim)
        """
        return self.current_opponent.act(obs)

    def add_opponent(self, opponent: Opponent):
        self.win_rates[opponent.get_name()] = 1
        
        self.opponents.append(opponent)

    def new_opponent(self):
        weights = []

        for opponent_name, p in self.win_rates.items():
            pfsp_w = p * (1 - p)
            hard_w = (1 - p) * 3
            radnom_w = 1

            weights.append(pfsp_w + hard_w + radnom_w)
        
        self.current_opponent = random.choices(self.opponents, weights=weights, k=1)[0]
        print(f"New opponent: {self.current_opponent.get_name()}")
    
    def get_opponent_name(self):
        return self.current_opponent.get_name()
    
    def calculate_matchmaking(self, current_agent: Opponent):
        for opponent in self.opponents:
            wins, losses, draws = play(
                player=current_agent,
                opponent=opponent,
                games=100,
                headless=True
            )
            win_rate = wins / (wins + losses + draws)
            self.win_rates[opponent.get_name()] = win_rate

        print(self.win_rates)



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
        self.scores = {}
        
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
        self.scores[opponent.get_name()] = {}
        for opponent2 in self.opponents:
            if opponent.get_name() != opponent2.get_name():
                wins, losses, draws = play(
                    player=opponent,
                    opponent=opponent2,
                    games=100,
                    headless=True
                )
                win_rate = wins / (wins + losses + draws)
                losse_rate = losses / (wins + losses + draws)
                self.scores[opponent.get_name()][opponent2.get_name()] = win_rate

                if opponent2.get_name() in self.scores:
                    self.scores[opponent2.get_name()][opponent.get_name()] = losse_rate

        self.opponents.append(opponent)

        print(self.scores)

    def new_opponent(self):
        self.current_opponent = random.choice(self.opponents)
        print(f"New opponent: {self.current_opponent.get_name()}")
    
    def get_opponent_name(self):
        return self.current_opponent.get_name()

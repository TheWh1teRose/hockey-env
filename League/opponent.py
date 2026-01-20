import torch
import numpy as np

class Opponent:
    def __init__(self, name: str):
        self.name = name

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def get_name(self) -> str:
        return self.name
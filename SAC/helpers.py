import numpy as np

def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """
    Normalize observations.
    
    Supports both single observations (obs_dim,) and batched observations (batch_size, obs_dim).
    
    Args:
        obs: Observation(s) with shape (obs_dim,) or (batch_size, obs_dim)
        
    Returns:
        Normalized observations with same shape as input
    """
    scaling = np.array([ 1.0,  1.0 , 0.5, 4.0, 4.0, 4.0,  
            1.0,  1.0,  0.5, 4.0, 4.0, 4.0,  
            2.0, 2.0, 10.0, 10.0, 4.0 ,4.0])

    return obs / scaling

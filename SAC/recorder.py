"""
WandB Recorder for SAC Training

This module provides a comprehensive logging and checkpoint management system
using Weights & Biases for monitoring SAC training on the Hockey environment.

The recorder is designed to be easily extensible through custom metric loggers.

created with Opus 4.5with prompt: Write me an recorder class that logs all important metrics to debug and supervice the learning process in weights and biases. Also the model checkpoints should be backupted to weights and biases. The class should be easely extendabile. 
"""

import os
import wandb
import torch
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime


class WandBRecorder:
    """
    A comprehensive Weights & Biases recorder for SAC training.
    
    Handles logging of training metrics, episode statistics, and model checkpoints
    with support for custom metric loggers for extensibility.
    
    Attributes:
        project (str): WandB project name
        run_name (str): Name of the current run
        checkpoint_dir (Path): Directory for saving checkpoints locally
        custom_loggers (dict): Registered custom metric loggers
    """
    
    def __init__(
        self,
        project: str,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        checkpoint_freq: int = 100,
        save_best: bool = True,
        save_local: bool = True,
        save_wandb: bool = True,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        entity: Optional[str] = None,
    ):
        """
        Initialize the WandB recorder.
        
        Args:
            project: WandB project name
            config: Hyperparameter configuration dict to log
            run_name: Optional name for this run (auto-generated if None)
            checkpoint_dir: Local directory for saving checkpoints
            checkpoint_freq: Save checkpoint every N episodes
            save_best: Whether to save checkpoint on best performance
            save_local: Whether to save checkpoints locally
            save_wandb: Whether to upload checkpoints to WandB
            tags: Optional list of tags for the run
            notes: Optional notes/description for the run
            mode: WandB mode ('online', 'offline', 'disabled')
            entity: WandB entity (username or team name)
        """
        self.project = project
        self.checkpoint_freq = checkpoint_freq
        self.save_best = save_best
        self.save_local = save_local
        self.save_wandb = save_wandb
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"sac_{timestamp}"
        self.run_name = run_name
        
        # Create checkpoint directory with run name
        self.checkpoint_dir = Path(checkpoint_dir) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize WandB
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=tags,
            notes=notes,
            mode=mode,
            entity=entity,
            reinit=True,
        )
        
        # Track best metrics for saving best model
        self.best_reward = float('-inf')
        self.best_win_rate = 0.0
        
        # Tracking counters
        self.episode_count = 0
        self.update_count = 0
        
        # Rolling stats for smoothing
        self._recent_rewards: List[float] = []
        self._recent_wins: List[int] = []
        self._rolling_window = 100
        
        # Custom loggers for extensibility
        self.custom_loggers: Dict[str, Callable] = {}
        
        # Define metrics for better visualization in WandB
        wandb.define_metric("train/*", step_metric="episode")
        wandb.define_metric("episode/*", step_metric="episode")
        wandb.define_metric("buffer/*", step_metric="episode")
        wandb.define_metric("system/*", step_metric="episode")
    
    def log_step(
        self,
        episode: int,
        metrics: Dict[str, Any],
        prefix: str = "step"
    ) -> None:
        """
        Log per-step metrics (e.g., action statistics, observations).
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names
        """
        self.episode_count = episode
        
        log_dict = {"episode": episode}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                log_dict[f"{prefix}/{key}"] = value
            elif isinstance(value, np.ndarray):
                # Log array statistics
                log_dict[f"{prefix}/{key}_mean"] = value.mean()
                log_dict[f"{prefix}/{key}_std"] = value.std()
        
        # Apply custom loggers
        for name, logger_fn in self.custom_loggers.items():
            try:
                custom_metrics = logger_fn(episode, metrics)
                if custom_metrics:
                    log_dict.update(custom_metrics)
            except Exception as e:
                print(f"Warning: Custom logger '{name}' failed: {e}")
        
        wandb.log(log_dict)
    
    def log_update(
        self,
        episode: int,
        actor_loss: float,
        critic_loss: float,
        alpha_loss: float,
        alpha: Optional[float] = None,
        q1_mean: Optional[float] = None,
        q2_mean: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log SAC update losses and related metrics.
        
        Args:
            global_step: Current global step counter
            actor_loss: Actor network loss
            critic_loss: Critic network loss
            alpha_loss: Entropy coefficient loss
            alpha: Current entropy coefficient value
            q1_mean: Mean Q1 value (optional)
            q2_mean: Mean Q2 value (optional)
            extra_metrics: Additional metrics to log
        """
        self.episode_count = episode
        self.update_count += 1
        
        log_dict = {
            "episode": episode,
            "train/actor_loss": actor_loss,
            "train/critic_loss": critic_loss,
            "train/alpha_loss": alpha_loss,
            "train/update_count": self.update_count,
        }
        
        if alpha is not None:
            log_dict["train/alpha"] = alpha if isinstance(alpha, float) else alpha.item()
        
        if q1_mean is not None:
            log_dict["train/q1_mean"] = q1_mean
        
        if q2_mean is not None:
            log_dict["train/q2_mean"] = q2_mean
        
        if extra_metrics:
            for key, value in extra_metrics.items():
                log_dict[f"train/{key}"] = value
        
        wandb.log(log_dict)
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        winner: int = 0,
        info: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log episode summary metrics.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length (number of steps)
            winner: Game outcome (1=win, -1=loss, 0=draw)
            info: Environment info dict with proxy rewards
            extra_metrics: Additional metrics to log
        """
        self.episode_count = episode
        
        # Update rolling stats
        self._recent_rewards.append(reward)
        self._recent_wins.append(1 if winner == 1 else 0)
        
        if len(self._recent_rewards) > self._rolling_window:
            self._recent_rewards.pop(0)
            self._recent_wins.pop(0)
        
        # Calculate rolling statistics
        rolling_reward = np.mean(self._recent_rewards)
        rolling_win_rate = np.mean(self._recent_wins) if self._recent_wins else 0.0
        
        log_dict = {
            "episode": episode,
            "episode/reward": reward,
            "episode/length": length,
            "episode/winner": winner,
            "episode/rolling_reward": rolling_reward,
            "episode/rolling_win_rate": rolling_win_rate,
        }
        
        # Log environment info metrics if provided
        if info:
            if "reward_closeness_to_puck" in info:
                log_dict["episode/closeness_to_puck"] = info["reward_closeness_to_puck"]
            if "reward_touch_puck" in info:
                log_dict["episode/touch_puck"] = info["reward_touch_puck"]
            if "reward_puck_direction" in info:
                log_dict["episode/puck_direction"] = info["reward_puck_direction"]
        
        if extra_metrics:
            for key, value in extra_metrics.items():
                log_dict[f"episode/{key}"] = value
        
        wandb.log(log_dict)
        
        # Check for best model
        if self.save_best:
            if rolling_reward > self.best_reward and len(self._recent_rewards) >= 10:
                self.best_reward = rolling_reward
                return True  # Signal that this is best so far
        
        return False
    
    def log_buffer(
        self,
        episode: int,
        buffer_size: int,
        buffer_capacity: int,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log replay buffer statistics.
        
        Args:
            global_step: Current global step counter
            buffer_size: Current number of transitions in buffer
            buffer_capacity: Maximum buffer capacity
            extra_metrics: Additional buffer metrics (e.g., priority stats)
        """
        log_dict = {
            "episode": episode,
            "buffer/size": buffer_size,
            "buffer/capacity": buffer_capacity,
            "buffer/fill_ratio": buffer_size / buffer_capacity,
        }
        
        if extra_metrics:
            for key, value in extra_metrics.items():
                log_dict[f"buffer/{key}"] = value
        
        wandb.log(log_dict)
    
    def save_checkpoint(
        self,
        agent,
        step: int,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a model checkpoint locally and/or upload to WandB artifacts.
        
        Args:
            agent: The SAC agent with a save() method
            step: Current step/episode number
            is_best: Whether this is the best model so far
            metadata: Additional metadata to store with checkpoint
            
        Returns:
            Path to the saved checkpoint file
        """
        # Create checkpoint filename
        if is_best:
            filename = f"checkpoint_best.pt"
        else:
            filename = f"checkpoint_step_{step}.pt"
        
        filepath = self.checkpoint_dir / filename
        
        # Save the model locally
        if self.save_local:
            agent.save(str(filepath))
            print(f"Checkpoint saved locally: {filepath}")
        
        # Upload to WandB
        if self.save_wandb:
            # If not saving locally, save to a temp location for wandb upload
            if not self.save_local:
                import tempfile
                temp_dir = Path(tempfile.mkdtemp())
                filepath = temp_dir / filename
                agent.save(str(filepath))
            
            # Create wandb artifact
            artifact_name = f"{self.run_name}_checkpoint"
            if is_best:
                artifact_name += "_best"
            
            # Sanitize best_reward for JSON serialization (handle -inf, inf, nan)
            best_reward_json = self.best_reward if np.isfinite(self.best_reward) else None
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"SAC checkpoint at step {step}",
                metadata={
                    "step": step,
                    "episode": self.episode_count,
                    "is_best": is_best,
                    "best_reward": best_reward_json,
                    **(metadata or {}),
                }
            )
            
            artifact.add_file(str(filepath))
            wandb.log_artifact(artifact)
            print(f"Checkpoint uploaded to WandB: {artifact_name}")
        
        return str(filepath)
    
    def add_logger(
        self,
        name: str,
        logger_fn: Callable[[int, Dict[str, Any]], Optional[Dict[str, Any]]]
    ) -> None:
        """
        Register a custom metric logger for extensibility.
        
        The logger function receives (episode, data) and should return
        a dict of metrics to log, or None.
        
        Args:
            name: Unique name for the logger
            logger_fn: Function that processes data and returns metrics dict
            
        Example:
            recorder.add_logger(
                "action_stats",
                lambda episode, data: {
                    "custom/action_entropy": compute_entropy(data["actions"])
                }
            )
        """
        self.custom_loggers[name] = logger_fn
        print(f"Registered custom logger: {name}")
    
    def remove_logger(self, name: str) -> bool:
        """
        Remove a registered custom logger.
        
        Args:
            name: Name of the logger to remove
            
        Returns:
            True if logger was removed, False if not found
        """
        if name in self.custom_loggers:
            del self.custom_loggers[name]
            return True
        return False
    
    def log_histogram(
        self,
        name: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        episode: Optional[int] = None,
    ) -> None:
        """
        Log a histogram of values.
        
        Args:
            name: Name for the histogram
            values: Array of values to create histogram from
            episode: Optional episode value (uses episode_count if not provided)
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, list):
            values = np.array(values)
        
        wandb.log({
            name: wandb.Histogram(values),
            "episode": episode or self.episode_count,
        })
    
    def log_video(
        self,
        name: str,
        frames: np.ndarray,
        fps: int = 30,
        episode: Optional[int] = None,
    ) -> None:
        """
        Log a video from frames.
        
        Args:
            name: Name for the video
            frames: Array of frames (T, H, W, C) or (T, C, H, W)
            fps: Frames per second
            episode: Optional episode value
        """
        # Ensure frames are in (T, H, W, C) format
        if frames.shape[1] in [1, 3, 4]:  # Likely (T, C, H, W)
            frames = np.transpose(frames, (0, 2, 3, 1))
        
        wandb.log({
            name: wandb.Video(frames, fps=fps, format="mp4"),
            "episode": episode or self.episode_count,
        })
    
    def watch_model(
        self,
        model: torch.nn.Module,
        log: str = "gradients",
        log_freq: int = 1000,
    ) -> None:
        """
        Watch a PyTorch model for gradient/parameter logging.
        
        Args:
            model: PyTorch model to watch
            log: What to log ('gradients', 'parameters', 'all')
            log_freq: How often to log
        """
        wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self, quiet: bool = False) -> None:
        """
        Finish the WandB run and cleanup.
        
        Args:
            quiet: Whether to suppress finish message
        """
        if not quiet:
            print(f"Finishing WandB run: {self.run_name}")
            print(f"Total episodes: {self.episode_count}")
            print(f"Best rolling reward: {self.best_reward:.2f}")
        
        wandb.finish()
    
    @property
    def url(self) -> Optional[str]:
        """Get the WandB run URL."""
        return self.run.url if self.run else None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures finish is called."""
        self.finish(quiet=exc_type is not None)
        return False

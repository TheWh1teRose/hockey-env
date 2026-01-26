#!/usr/bin/env python3
"""
Play trained SAC checkpoints in the Hockey environment.
Runs in an endless loop with rendering enabled.

created with Opus 4.5 with prompt: pls create a play.py that can play specific trained checkpoints in an edless loop.
"""

import argparse
import yaml
import numpy as np
import torch
import hockey.hockey_env as h_env
from SAC.models import Actor
from League.opponent import Opponent
from League.opponents.self import SelfPlayOpponent
from League.opponents.handcrafted import HandcraftedOpponent


def load_actor(checkpoint_path: str, state_dim: int, action_dim: int, hidden_dim: int, device: str) -> Actor:
    """Load only the actor network from a checkpoint."""
    actor = Actor(state_dim, action_dim, hidden_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    return actor


def play(
    device: str = "cpu",
    player: Opponent = None,
    opponent: Opponent = None,
    games: int = 50,
    headless: bool = False
):
    """Play the hockey environment with a trained agent in an endless loop."""
    
    # Initialize environment
    env = h_env.HockeyEnv()
    
    state_dim = env.observation_space.shape[0]  # 18
    action_dim = 4  # x, y, angle, shoot
    
    episode = 0
    wins = 0
    losses = 0
    draws = 0
    
    try:
        for i in range(games):
            episode += 1
            obs, info = env.reset()
            
            done = False
            episode_reward = 0
            step = 0
            
            while not done:
                # Render the environment
                if not headless:
                    env.render()
                
                # Get agent action
                action_1 = player.act(obs)
                
                # Get opponent action
                obs_agent2 = env.obs_agent_two()
                action_2 = opponent.act(obs_agent2)
                
                # Combine actions and step
                env_action = np.hstack([action_1, action_2])
                next_obs, reward, terminated, truncated, info = env.step(env_action)
                
                episode_reward += reward
                obs = next_obs
                step += 1
                
                done = terminated or truncated
            
            # Track results
            winner = info.get("winner", 0)
            if winner == 1:
                wins += 1
                result = "WIN"
            elif winner == -1:
                losses += 1
                result = "LOSS"
            else:
                draws += 1
                result = "DRAW"
            
            # Print episode summary
    finally:
        env.close()

    return wins, losses, draws


def main():
    parser = argparse.ArgumentParser(
        description="Play trained SAC checkpoints in the Hockey environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the checkpoint file (.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension of the actor network (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run inference on (overrides config)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        choices=["basic_weak", "basic_strong", "human"],
        help="Type of opponent to play against (overrides config)"
    )
    parser.add_argument(
        "--headless",
        type=bool,
        default=False,
        help="Run in headless mode"

    )

    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get settings from config, allow CLI args to override
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else config["sac"]["hidden_dim"]
    device = args.device if args.device is not None else config["training"]["device"]
    
    # Parse opponent from config if not specified via CLI
    if args.opponent is not None:
        opponent_type = args.opponent
    else:
        config_opponent = config["init_opponents"][0]
        if config_opponent["type"] == "checkpoint":
            opponent_type = "self"
            opponent = SelfPlayOpponent(
                name="opponent", 
                model_path=config_opponent["model_path"], 
                device=config["training"]["device"]
            )
        elif config_opponent["type"] == "handcrafted":
            if config_opponent["strength"] == "weak":
                opponent_type = "basic_weak"
                opponent = HandcraftedOpponent("opponent", "weak")
            elif config_opponent["strength"] == "strong":
                opponent_type = "basic_strong"
                opponent = HandcraftedOpponent("opponent", "strong")
        else:
            raise ValueError(f"Unknown opponent type: {config_opponent['type']}")

    player = SelfPlayOpponent(
        name="player", 
        model_path=args.checkpoint, 
        device=config["training"]["device"]
    )

    
    
    wins, losses, draws = play(
        device=device,
        player=player,
        opponent=opponent,
        headless=args.headless
    )

    
    # Final stats
    total_games = wins + losses + draws
    if total_games > 0:
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total Episodes: {total_games}")
        print(f"Wins:   {wins:4d} ({wins/total_games:.1%})")
        print(f"Losses: {losses:4d} ({losses/total_games:.1%})")
        print(f"Draws:  {draws:4d} ({draws/total_games:.1%})")
        print("="*50)


if __name__ == "__main__":
    main()

import yaml
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
from SAC.replay_buffer import PrioritizedReplayBuffer
from SAC.SAC import SAC
from SAC.recorder import WandBRecorder
import hockey.hockey_env as h_env
from SAC.helpers import normalize_obs

def train(config, checkpoint=None):

    env = h_env.HockeyEnv()

    buffer = PrioritizedReplayBuffer(config["buffer"]["size"], env.observation_space.shape[0], 4)
    sac = SAC(buffer, env.observation_space.shape[0], 4, config["sac"]["hidden_dim"], 
            config["sac"]["lr"], config["sac"]["gamma"], config["sac"]["tau"], 
            config["sac"]["alpha"], config["training"]["device"])

    opponent = h_env.BasicOpponent(weak=False)

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"training_{time_stamp}"

    recorder = WandBRecorder(
        project="hockey-sac",
        config=config,
        run_name=run_name,
        checkpoint_dir="checkpoints",
        checkpoint_freq=config["training"]["checkpoint_freq"],
        save_best=False,
        tags=["sac", "training"],
        notes="SAC training",
    )

    obs, info = env.reset(seed=42)
    obs = normalize_obs(obs)

    global_step = 0
    total_reward = 0

    wins = 0
    losses = 0
    draws = 0

    for episode in range(config["training"]["num_episodes"]):
        episode_length = 0
        last_info = info

        # Accumulators for training metrics
        episode_metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "alpha": [],
            "q1_mean": [],
            "q2_mean": [],
        }
        
        # Accumulators for environment info rewards
        episode_info_rewards = {
            "reward_closeness_to_puck": 0.0,
            "reward_touch_puck": 0.0,
            "reward_puck_direction": 0.0,
        }

        for step in range(config["training"]["episode_length"]):

            global_step += 1
            episode_length += 1

            env.render()

            action_1 = sac.act(obs)

            # Get opponent action
            obs_agent2 = env.obs_agent_two()
            action_2 = opponent.act(obs_agent2)

            env_action = np.hstack([action_1, action_2])

            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            next_obs = normalize_obs(next_obs)
            total_reward += reward

            # Accumulate info rewards
            for key in episode_info_rewards:
                if key in info:
                    episode_info_rewards[key] += info[key]

            buffer.add(obs, action_1, reward, next_obs, terminated)
            obs = next_obs

            # Update SAC and accumulate metrics
            if buffer.is_ready(config["buffer"]["min_size"]):
                metrics = sac.update()
                for key in episode_metrics:
                    episode_metrics[key].append(metrics[key])

            if terminated or truncated:

                if last_info.get("winner", 0) == 1:
                    wins += 1
                elif last_info.get("winner", 0) == -1:
                    losses += 1
                else:
                    draws += 1

                obs, info = env.reset()
                obs = normalize_obs(obs)
        
        # Log accumulated training metrics at end of episode
        if episode_metrics["actor_loss"]:  # Only log if we had updates
            recorder.log_update(
                episode=episode + 1,
                actor_loss=np.mean(episode_metrics["actor_loss"]),
                critic_loss=np.mean(episode_metrics["critic_loss"]),
                alpha_loss=np.mean(episode_metrics["alpha_loss"]),
                alpha=np.mean(episode_metrics["alpha"]),
                q1_mean=np.mean(episode_metrics["q1_mean"]),
                q2_mean=np.mean(episode_metrics["q2_mean"]),
                extra_metrics={
                    "num_updates": len(episode_metrics["actor_loss"]),
                }
            )

        # Log episode metrics with accumulated info rewards
        is_best = recorder.log_episode(
            episode=episode + 1,
            reward=total_reward,
            length=episode_length,
            winner=last_info.get("winner", 0),
            info=episode_info_rewards,  # Pass accumulated rewards instead of last_info
            extra_metrics={
                "wins_total": wins,
                "losses_total": losses,
                "draws_total": draws,
                "win_rate": wins / (episode + 1),
            }
        )

        # Log buffer stats periodically
        if (episode + 1) % 10 == 0:
            recorder.log_buffer(episode + 1, len(buffer), buffer.capacity)
        
        # Save checkpoint
        if (episode + 1) % config["training"]["checkpoint_freq"] == 0:
            recorder.save_checkpoint(sac, episode + 1, is_best=is_best)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} | Reward: {total_reward:.2f} | "
                  f"W/L/D: {wins}/{losses}/{draws} | Win Rate: {wins/(episode+1):.2%}")
        
        total_reward = 0
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    train(config, args.checkpoint)

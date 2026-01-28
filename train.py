import yaml
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from datetime import datetime
from SAC.replay_buffer import PrioritizedReplayBuffer
from SAC.SAC import SAC
from SAC.recorder import WandBRecorder
import hockey.hockey_env as h_env
from SAC.helpers import normalize_obs
from League.league import League
from League.opponents.self import SelfPlayOpponent
import copy
import threading


def make_env():
    """Factory function to create a HockeyEnv instance."""
    def _init():
        return h_env.HockeyEnv()
    return _init

def train(config, checkpoint=None):
    
    device = config["training"]["device"]
    num_envs = config["training"]["num_envs"]
    
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = 4

    buffer = PrioritizedReplayBuffer(config["buffer"]["size"], obs_dim, action_dim, device=device)
    
    gpu_config = config.get("gpu_optimization", {})
    sac = SAC(
        buffer, obs_dim, action_dim, 
        config["sac"]["hidden_dim"], 
        config["sac"]["lr"], 
        config["sac"]["gamma"], 
        config["sac"]["tau"], 
        config["sac"]["alpha"], 
        device,
        batch_size=config["buffer"].get("batch_size", 512),
        use_amp=gpu_config.get("use_amp", True),
        use_compile=gpu_config.get("use_compile", True),
        updates_per_step=gpu_config.get("updates_per_step", 4)
    )

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")
        sac.load(checkpoint)

    league = League(config)

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
        notes="SAC training with parallel environments",
    )

    # start exploiter training thread
    exploiter_running = threading.Event()
    exploiter_running.set()

    main_episodes = [0] # shared variable to track main episodes

    exploiter_thread = threading.Thread(
        target=train_expoiter,
        args=(config, sac, league, recorder, main_episodes, exploiter_running),
        daemon=True
    )
    exploiter_thread.start()

    obs, info = envs.reset()
    obs = normalize_obs(obs)

    env_rewards = np.zeros(num_envs)
    env_lengths = np.zeros(num_envs, dtype=int)

    # Global statistics
    total_games = 0
    wins = 0
    losses = 0
    draws = 0

    for episode in range(config["training"]["num_episodes"]):
        main_episodes[0] = episode
        episode_total_reward = 0.0
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
            episode_length += 1

            if not config["environment"]["headless"]:
                envs.render()

            action_1 = sac.act(obs)

            obs_agent2 = np.array(envs.call("obs_agent_two"))
            
            action_2 = league.act(obs_agent2)

            env_action = np.hstack([action_1, action_2])

            next_obs, rewards, terminated, truncated, infos = envs.step(env_action)
            next_obs = normalize_obs(next_obs)
            
            env_rewards += rewards
            env_lengths += 1
            episode_total_reward += rewards.sum()

            for key in episode_info_rewards:
                if key in infos:
                    episode_info_rewards[key] += np.sum(infos[key])

            buffer.add_batch(obs, action_1, rewards, next_obs, terminated)
            obs = next_obs

            # Update SAC and accumulate metrics
            if buffer.is_ready(config["buffer"]["min_size"]):
                metrics = sac.update()
                for key in episode_metrics:
                    episode_metrics[key].append(metrics[key])

            # Handle episode terminations for each env
            done_mask = terminated | truncated
            if done_mask.any():
                for env_idx in np.where(done_mask)[0]:
                    total_games += 1
                    
                    winner = infos.get("winner", np.zeros(num_envs))[env_idx]
                    if winner == 1:
                        wins += 1
                    elif winner == -1:
                        losses += 1
                    else:
                        draws += 1
                    
                    env_rewards[env_idx] = 0
                    env_lengths[env_idx] = 0
 
                last_info = infos

        ### End of Episode ###

        # Log accumulated training metrics
        if episode_metrics["actor_loss"]:
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
            opponent=league.get_opponent_name(),
            reward=episode_total_reward,
            length=episode_length * num_envs,
            winner=last_info.get("winner", np.zeros(num_envs))[0] if isinstance(last_info.get("winner", 0), np.ndarray) else last_info.get("winner", 0),
            info=episode_info_rewards,
            extra_metrics={
                "wins_total": wins,
                "losses_total": losses,
                "draws_total": draws,
                "win_rate": wins / max(total_games, 1),
                "num_envs": num_envs,
                "total_games": total_games,
            }
        )

        if episode % config["training"]["cal_matchmaking_freq"] == 0 and episode > 0:
            actor = copy.deepcopy(sac.actor)
            league.calculate_matchmaking(SelfPlayOpponent(name=f"opponent_{episode}", Actor=actor, device=device))

        # Reset all envs at end of episode
        obs, info = envs.reset()
        obs = normalize_obs(obs)
        env_rewards[:] = 0
        env_lengths[:] = 0
        
        league.new_opponent()

        if episode % config["training"]["add_opponent_freq"] == 0 and episode > 0:
            print(f"Adding opponent {episode}")
            actor = copy.deepcopy(sac.actor)
            league.add_opponent(SelfPlayOpponent(name=f"opponent_{episode}", Actor=actor, device=device))

        total_games = 0
        wins = 0
        looses = 0
        draws = 0

        # Log buffer stats periodically
        if (episode + 1) % 10 == 0:
            recorder.log_buffer(episode + 1, len(buffer), buffer.capacity)
        
        # Save checkpoint
        if (episode + 1) % config["training"]["checkpoint_freq"] == 0:
            recorder.save_checkpoint(sac, episode + 1, is_best=is_best)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} | Reward: {episode_total_reward:.2f} | "
                  f"W/L/D: {wins}/{losses}/{draws} | Win Rate: {wins/max(total_games, 1):.2%} | "
                  f"Games: {total_games}")

    exploiter_running.clear()
    exploiter_thread.join()
    
    envs.close()


def train_expoiter(config, main_agent_sac, league, recorder, main_episodes, running):
    num_envs = config["training"]["exploiter"]["num_envs"]
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = 4

    device = config["training"]["device"]

    buffer = PrioritizedReplayBuffer(config["buffer"]["size"], obs_dim, action_dim, device=device)
    gpu_config = config.get("gpu_optimization", {})
    sac = SAC(
        buffer, obs_dim, action_dim, 
        config["sac"]["hidden_dim"], 
        config["sac"]["lr"], 
        config["sac"]["gamma"], 
        config["sac"]["tau"], 
        config["sac"]["alpha"], 
        device,
        batch_size=config["buffer"].get("batch_size", 512),
        use_amp=gpu_config.get("use_amp", True),
        use_compile=gpu_config.get("use_compile", True),
        updates_per_step=gpu_config.get("updates_per_step", 4)
    )

    # global statistics
    total_games = 0
    wins = 0
    losses = 0
    draws = 0
    episode = 0

    # Per-env tracking
    env_rewards = np.zeros(num_envs)
    env_lengths = np.zeros(num_envs, dtype=int)

    obs, info = envs.reset()
    obs = normalize_obs(obs)

    print(f"Exploiter training started")

    while running.is_set():
        episode += 1

        episode_total_reward = 0.0
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

        for step in range(config["training"]["exploiter"]["episode_length"]):
            episode_length += 1

            action_1 = sac.act(obs)

            obs_agent2 = np.array(envs.call("obs_agent_two"))
            obs_agent2 = normalize_obs(obs_agent2)
            action_2 = main_agent_sac.act(obs_agent2)

            env_action = np.hstack([action_1, action_2])
            next_obs, rewards, terminated, truncated, infos = envs.step(env_action)
            next_obs = normalize_obs(next_obs)

            env_rewards += rewards
            env_lengths += 1
            episode_total_reward += rewards.sum()

            for key in episode_info_rewards:
                if key in infos:    
                    episode_info_rewards[key] += np.sum(infos[key])

            buffer.add_batch(obs, action_1, rewards, next_obs, terminated)
            obs = next_obs

            # Update SAC and accumulate metrics
            if buffer.is_ready(config["buffer"]["min_size"]):
                metrics = sac.update()
                for key in episode_metrics:
                    episode_metrics[key].append(metrics[key])

            # Handle episode terminations for each env
            done_mask = terminated | truncated
            if done_mask.any():
                for env_idx in np.where(done_mask)[0]:
                    total_games += 1
                    winner = infos.get("winner", np.zeros(num_envs))[env_idx]

                    if winner == 1:
                        wins += 1
                    elif winner == -1:
                        losses += 1
                    else:
                        draws += 1

                    env_rewards[env_idx] = 0
                    env_lengths[env_idx] = 0

                last_info = infos

        ### End of Episode ###

        # Log accumulated training metrics for exploiter
        if episode_metrics["actor_loss"]:
            recorder.log_exploiter_update(
                episode=episode,
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

        # Log episode metrics for exploiter
        recorder.log_exploiter_episode(
            episode=episode,
            reward=episode_total_reward,
            length=episode_length * num_envs,
            winner=last_info.get("winner", np.zeros(num_envs))[0] if isinstance(last_info.get("winner", 0), np.ndarray) else last_info.get("winner", 0),
            info=episode_info_rewards,
            extra_metrics={
                "wins_total": wins,
                "losses_total": losses,
                "draws_total": draws,
                "win_rate": wins / max(total_games, 1),
                "loss_rate": losses / max(total_games, 1),
                "draw_rate": draws / max(total_games, 1),
                "num_envs": num_envs,
                "total_games": total_games,
                "main_agent_episode": main_episodes,
            }
        )

        # Reset all envs at end of episode
        obs, info = envs.reset()
        obs = normalize_obs(obs)
        env_rewards[:] = 0
        env_lengths[:] = 0

        # if winrate is over 0.7 add exploiter to league and reset exploiter
        if wins / max(total_games, 1) > 0.7 and main_episodes > 20:
            actor = copy.deepcopy(sac.actor)
            league.add_opponent(SelfPlayOpponent(name=f"exploiter_{episode}", Actor=actor, device=device))

            print(f"Exploiter {episode} added to league")

            sac.save(f"checkpoints/exploiter_{episode}.pt")
            print(f"Exploiter {episode} checkpoint saved")

            total_games = 0
            wins = 0
            losses = 0
            draws = 0
            sac = SAC(buffer, obs_dim, action_dim, config["sac"]["hidden_dim"], config["sac"]["lr"], config["sac"]["gamma"], config["sac"]["tau"], config["sac"]["alpha"], device, batch_size=config["buffer"].get("batch_size", 512), use_amp=gpu_config.get("use_amp", True), use_compile=gpu_config.get("use_compile", True), updates_per_step=gpu_config.get("updates_per_step", 4))

        # Reset per-episode stats
        total_games = 0
        wins = 0
        losses = 0
        draws = 0

    envs.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    train(config, args.checkpoint)

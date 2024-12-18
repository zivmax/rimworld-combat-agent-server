import gymnasium as gym
import os
import logging
from tqdm import tqdm
from datetime import datetime
from gymnasium.wrappers import RecordEpisodeStatistics

from agents.dqn import DQNAgent
from env import rimworld_env
from utils.logger import get_file_logger
from utils.draw import draw
from .hyper_params import N_EPISODES, EPISOLD_LOG_INTERVAL, EPISOLD_SAVE_INTERVAL
from .hyper_params import RE_TRAIN, OPTIONS, LOAD_PATH


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
f_logger = get_file_logger(f"agents/dqn/logs/{timestamp}.log", logging.DEBUG)


def main():
    """
    Main function to train the DQNAgent.
    """

    env = gym.make(rimworld_env, options=OPTIONS)
    env = RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    if RE_TRAIN:

        save_dir = f"agents/dqn/models/{timestamp}"

        num_episodes = N_EPISODES
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.act(obs, info)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.step(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

            if (episode + 1) % EPISOLD_LOG_INTERVAL == 0:
                f_logger.debug(f"\tFor episode {episode + 1}, reward: {total_reward}")

            if (episode + 1) % EPISOLD_SAVE_INTERVAL == 0:
                agent.save(os.path.join(save_dir, f"episode_{episode + 1}.pth"))
    else:
        agent.load(LOAD_PATH)
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        f_logger.info(f"\tTotal reward after loading: {total_reward}")
    env.close()

    draw(env, "agents/dqn/plot/episode_stats.png")


if __name__ == "__main__":
    main()

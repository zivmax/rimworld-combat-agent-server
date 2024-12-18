import gymnasium as gym
import os
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics

from agents.dqn import DQNAgent
from env import rimworld_env
from utils.draw import draw
from utils.timestamp import timestamp
from utils.json import to_json

from .logger import logger
from .hyper_params import N_EPISODES, EPISOLD_LOG_INTERVAL, EPISOLD_SAVE_INTERVAL
from .hyper_params import RE_TRAIN, OPTIONS, LOAD_PATH


def main():
    """
    Main function to train the DQNAgent.
    """
    logger.info("Configs: \n" + to_json(OPTIONS))
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
                logger.debug(f"\tFor episode {episode + 1}, reward: {total_reward}")

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
        logger.info(f"\tTotal reward after loading: {total_reward}")
    env.close()

    draw(env, "agents/dqn/plot/episode_stats.png")


if __name__ == "__main__":
    main()

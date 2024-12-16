import gymnasium as gym

from env import rimworld_env
from utils.logger import logger
from agents.random import RandomAgent


def main():
    env = gym.make(rimworld_env)
    agent = RandomAgent()
    n_episodes = 100

    for episode in range(n_episodes):
        obs, info = env.reset(options=agent.options)
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(obs, info)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward

        logger.info(
            f"\tEpisode {episode + 1}/{n_episodes}, Total Reward: {episode_reward}\n"
        )

    env.close()


if __name__ == "__main__":
    main()

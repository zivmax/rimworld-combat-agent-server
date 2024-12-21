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
from .hyper_params import (
    TRAINING,
    OPTIONS,
    LOAD_PATH,
    LOAD_TEST_EPISODES,
    CONTINUE_TRAINING_PATH,
    CONTINUE_NUM,
)


def main():
    """
    Main function to train the DQNAgent.
    """
    logger.info("\tConfigs: \n" + to_json(OPTIONS, indent=2))
    env = gym.make(rimworld_env, options=OPTIONS)
    env = RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    save_dir = f"agents/dqn/models/{timestamp}"

    num_episodes = N_EPISODES if TRAINING else LOAD_TEST_EPISODES
    if not TRAINING:
        agent.load(LOAD_PATH)
    if CONTINUE_TRAINING_PATH:
        agent.load(CONTINUE_TRAINING_PATH)
    for episode in tqdm(
        range(CONTINUE_NUM + 1, num_episodes + CONTINUE_NUM + 1),
        desc="Training Episodes" if TRAINING else "Testing Episodes",
    ):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if TRAINING:
                agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        if not TRAINING:
            logger.info(f"\tTotal reward for episode {episode + 1}: {total_reward}")
        if TRAINING and (episode + 1) % EPISOLD_LOG_INTERVAL == 0:
            logger.debug(f"\tTotal reward for episode {episode + 1}: {total_reward}")
        if TRAINING and (episode + 1) % EPISOLD_SAVE_INTERVAL == 0:
            agent.save(os.path.join(save_dir, f"episode_{episode + 1}.pth"))
            draw(env, f"agents/dqn/plots/episode_stats_{timestamp}.png")

    env.close()


if __name__ == "__main__":
    main()

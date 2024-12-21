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
    ENV_OPTIONS,
    LOAD_PATH,
    TEST_EPISODES,
    CONTINUE,
    CONTINUE_NUM,
)


def main():
    """
    Main function to train the DQNAgent.
    """
    logger.info("\tConfigs: \n" + to_json(ENV_OPTIONS, indent=2))
    env = gym.make(rimworld_env, options=ENV_OPTIONS)
    env = RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    starting_episode = 1
    save_dir = f"agents/dqn/models/{timestamp}"

    num_episodes = N_EPISODES if TRAINING else TEST_EPISODES

    if not TRAINING:
        agent.load(LOAD_PATH)
        if CONTINUE:
            starting_episode = CONTINUE_NUM

    for episode in tqdm(
        range(starting_episode, num_episodes + starting_episode),
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
            logger.info(f"\tTotal reward for episode {episode}: {total_reward}")
        if TRAINING and (episode) % EPISOLD_LOG_INTERVAL == 0:
            logger.debug(f"\tTotal reward for episode {episode}: {total_reward}")
        if TRAINING and (episode) % EPISOLD_SAVE_INTERVAL == 0:
            agent.save(os.path.join(save_dir, f"model_{episode}.pth"))
            draw(env, f"agents/dqn/plots/{timestamp}/stats_{episode}.png")

    env.close()


if __name__ == "__main__":
    main()

from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from agents.dqn import DQNAgent
from env import rimworld_env
from utils.logger import logger
from utils.draw import draw
from .hyper_params import N_EPISODES, EPISOLD_LOG_INTERVAL, EPISOLD_SAVE_INTERVAL
from .hyper_params import RE_TRAIN


OPTIONS = {
    "interval": 3.0,
    "speed": 4,
    "action_range": 4,
    "is_remote": False,
    "rewarding": {
        "original": 0,
        "ally_down": -10,
        "enemy_down": 10,
        "ally_danger_ratio": 0.5,
        "enemy_danger_ratio": -0.5,
    },
}


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
                logger.info(f"for episode {episode + 1}, reward: {total_reward}")

            if (episode + 1) % EPISOLD_SAVE_INTERVAL == 0:
                agent.save(f"agents/dqn/model_pth/dqn_model_episode_{episode + 1}.pth")
    else:
        agent.load(f"agents/dqn/model_pth/dqn_model_episode_{N_EPISODES}.pth")

    env.close()

    draw(env, "agents/dqn/plot/episode_stats.png")


if __name__ == "__main__":
    import logging

    file_handler = logging.FileHandler("agents/dqn/train.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s")
    )
    logger.addHandler(file_handler)
    main()

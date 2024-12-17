from agents.dqn import DQNAgent
from env import rimworld_env
from utils.logger import logger
import gymnasium as gym

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
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    if RE_TRAIN:
        num_episodes = N_EPISODES
        for episode in range(num_episodes):
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
                    logger.info(f"for episode {episode + 1}, reward: {reward}")

            logger.info(
                f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}"
            )

            if (episode + 1) % EPISOLD_SAVE_INTERVAL == 0:
                agent.save(f"./model_pth/dqn_model_episode_{episode + 1}.pth")
    else:
        agent.load(f"./model_pth/dqn_model_episode_{N_EPISODES}.pth")

    env.close()


if __name__ == "__main__":
    main()

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from env import rimworld_env
from agents.random import RandomAgent
from utils.draw import draw

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
    env = gym.make(rimworld_env, options=OPTIONS)
    env = RecordEpisodeStatistics(env, buffer_length=4)
    agent = RandomAgent(
        action_space=env.action_space, observation_space=env.observation_space
    )
    n_episodes = 3

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(obs, info)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward

    env.close()
    draw(env, "agents/random/plot/episode_stats.png")


if __name__ == "__main__":
    main()

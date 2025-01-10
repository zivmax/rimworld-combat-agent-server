import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from env import rimworld_env
from agents.random import RandomAgent
from utils.draw import draw


OPTIONS = {
    "interval": 0.5,
    "speed": 3,
    "action_range": 1,
    "max_steps": None,
    "is_remote": False,
    "remain_still_threshold": 300,
    "rewarding": {
        "original": 0,
        "win": 0,
        "lose": -0,
        "ally_defeated": -100,
        "enemy_defeated": 100,
        "ally_danger": -200,
        "enemy_danger": 200,
        "invalid_action": -0.25,
        "remain_still": -0.25,
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
        _, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act()
            _, reward, done, _, _ = env.step(action)
            episode_reward += reward

    env.close()
    draw(env, "agents/random/plot/episode_stats.png")


main()

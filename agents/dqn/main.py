import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics

from agents.dqn import DQNAgent as Agent
from env import rimworld_env
from utils.draw import draw
from utils.timestamp import timestamp

# this page is explicitly used fr storing trainning hyperparams
ENV_OPTIONS = {
    "interval": 0.5,
    "speed": 4,
    "action_range": 1,
    "is_remote": False,
    "rewarding": {
        "original": 0,
        "win": 100,
        "lose": -100,
        "ally_defeated": -10,
        "enemy_defeated": 10,
        "ally_danger": -10,
        "enemy_danger": 10,
    },
}


def main():
    n_episodes = 10000
    env = gym.make(rimworld_env, options=ENV_OPTIONS)
    env = RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = Agent(obs_space=env.observation_space, act_space=env.action_space[1])

    for episode in tqdm(range(1, n_episodes + 1), desc="Training Progress"):
        next_state, _ = env.reset()

        while True:
            current_state = next_state
            action = agent.act(current_state)
            action = {1: action}

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = next_obs

            agent.remember(current_state, next_state, action[1], reward, done)
            agent.train()

            if done:
                break

        if episode % 100 == 0 and episode > 0:
            agent.policy_net.save(f"./models/{timestamp}/dqn_{episode}.pth")
            draw(env, save_path=f"./plots/{timestamp}/stats_{episode}.png")

    env.close()


if __name__ == "__main__":
    main()

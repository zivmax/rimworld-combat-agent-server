import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, RecordEpisodeStatistics
from tqdm import tqdm

from agents.dqn import DQNAgent as Agent
from env import rimworld_env

ENV_OPTIONS = {
    "interval": 0.5,
    "speed": 1,
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

N_EPISODES = 2000
MODEL_PATH = "agents/dqn/models/2024-12-28_17:51:19/0700.pth"


def main():
    n_episodes = N_EPISODES
    env = gym.make(rimworld_env, options=ENV_OPTIONS)
    env = FrameStackObservation(env, stack_size=4)
    env = RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = Agent(obs_space=env.observation_space, act_space=env.action_space[1])
    agent.policy_net.load(MODEL_PATH)
    agent.epsilon_start = 0.001
    agent.epsilon_final = 0.001
    agent.epsilon_decay = 0
    agent.batch_size = 0
    agent.policy_net.eval()

    for episode in tqdm(range(1, n_episodes + 1), desc="Training Progress"):
        next_state, _ = env.reset()
        next_state.swapaxes(0, 1)
        while True:
            current_state = next_state
            action = agent.act(current_state)
            action = {1: action}

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = next_obs

            if done:
                break

    env.close()


if __name__ == "__main__":
    main()

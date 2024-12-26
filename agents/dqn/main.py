import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, FrameStackObservation

from agents.dqn import DQNAgent as Agent
from env import rimworld_env
from utils.draw import draw
from utils.timestamp import timestamp

ENV_OPTIONS = {
    "interval": 0.5,
    "speed": 2,
    "action_range": 1,
    "is_remote": False,
    "remain_still_threshold": 4,
    "rewarding": {
        "original": 0,
        "win": 0,
        "lose": -0,
        "ally_defeated": -10,
        "enemy_defeated": 10,
        "ally_danger": -10,
        "enemy_danger": 10,
        "invalid_action": -1,
        "remain_still": 0,
    },
}

N_EPISODES = 10000
SAVING_INTERVAL = 100


def main():
    n_episodes = N_EPISODES
    env = gym.make(rimworld_env, options=ENV_OPTIONS)
    env = FrameStackObservation(env, stack_size=4)
    env = RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = Agent(obs_space=env.observation_space, act_space=env.action_space[1])

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

            agent.remember(current_state, next_state, action[1], reward, done)
            agent.train()

            if done:
                break

        if episode % SAVING_INTERVAL == 0 and episode > 0:
            agent.policy_net.save(f"agents/dqn/models/{timestamp}/dqn_{episode}.pth")
            draw(env, save_path=f"agents/dqn/plots/{timestamp}/env_{episode}.png")

    env.close()


if __name__ == "__main__":
    main()

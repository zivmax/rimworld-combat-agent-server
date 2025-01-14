import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, FrameStackObservation
from agents.pgm import PGAgent as Agent
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

N_EPISODES = 1000
PLOT_INTERVAL = 100
MODEL_PATH = "agents/ppo/models/2024-12-28_19:19:28/ppo_800.pth"


def main():
    env = gym.make(rimworld_env, options=ENV_OPTIONS)
    env = FrameStackObservation(env, stack_size=4)
    env = RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    agent = Agent(obs_space=env.observation_space, act_space=env.action_space)
    agent.policy.load(MODEL_PATH)
    agent.policy.eval()
    episode_rewards = []
    for episode in tqdm(range(1, N_EPISODES + 1), desc="Evaluating Progress"):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(
                reward,
                next_state,
                done,
            )
            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)

        if episode % PLOT_INTERVAL == 0:
            draw(env, save_path=f"agents/ppo/plots/{timestamp}/env_{episode}.png")

    env.close()


if __name__ == "__main__":
    main()

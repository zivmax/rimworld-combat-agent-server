import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, RecordEpisodeStatistics
from tqdm import tqdm
import pandas as pd
import os

from agents.dqn import DQNAgent as Agent
from env import rimworld_env
from utils.draw import draw
from utils.timestamp import timestamp

ENV_OPTIONS = {
    "interval": 0.5,
    "speed": 4,
    "action_range": 1,
    "max_steps": 800,
    "is_remote": False,
    "remain_still_threshold": 100,
    "rewarding": {
        "original": 0,
        "win": 50,
        "lose": -50,
        "ally_defeated": -0,
        "enemy_defeated": 0,
        "ally_danger": -200,
        "enemy_danger": 200,
        "invalid_action": -0.25,
        "remain_still": 0,
    },
}

N_EPISODES = 20000
SAVING_INTERVAL = 500
MODEL_PATH = "agents/dqn/models/2025-01-02_13:45:44/10000.pth"


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
    agent.policy_net.train()

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
            agent.policy_net.save(f"agents/dqn/models/{timestamp}/{episode}.pth")
            agent.draw_model(f"agents/dqn/plots/training/{timestamp}/{episode}.png")
            agent.draw_agent(f"agents/dqn/plots/threshold/{timestamp}/{episode}.png")
            draw(env, save_path=f"agents/dqn/plots//env/{timestamp}/{episode}.png")
            saving(env, agent, timestamp, episode)

    env.close()


def saving(
    env: RecordEpisodeStatistics, agent: Agent, timestamp: str, episode: int
) -> None:
    # Saving all training history into csv

    # Create a DataFrame with the episode statistics
    eps_hist_df = pd.DataFrame(
        {
            "Episode": range(len(env.return_queue)),
            "Rewards": env.return_queue,
            "Length": env.length_queue,
            "Time": env.time_queue,
        }
    )

    # Create a DataFrame with the training statistics
    stats_df = pd.DataFrame(
        {
            "Update": range(len(agent.loss_history)),
            "Loss": agent.loss_history,
            "Q-Value": agent.q_value_history,
            "TD-Error": agent.td_error_history,
        }
    )

    # Create a DataFrame with the threshold history
    thres_df = pd.DataFrame(
        {
            "Steps": range(len(agent.eps_threshold_history)),
            "Threshold": agent.eps_threshold_history,
        }
    )

    os.makedirs(f"agents/dqn/history/{timestamp}/env/", exist_ok=True)
    os.makedirs(f"agents/dqn/history/{timestamp}/training/", exist_ok=True)
    os.makedirs(f"agents/dqn/history/{timestamp}/threshold/", exist_ok=True)

    eps_hist_df.to_csv(
        f"agents/dqn/history/{timestamp}/env/{episode}.csv",
        index=False,
    )
    stats_df.to_csv(
        f"agents/dqn/history/{timestamp}/training/{episode}.csv", index=False
    )
    thres_df.to_csv(
        f"agents/dqn/history/{timestamp}/threshold/{episode}.csv", index=False
    )


if __name__ == "__main__":
    main()

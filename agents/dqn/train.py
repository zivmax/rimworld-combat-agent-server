import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from agents.dqn import DQNAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers import (
    FrameStackObservation,
    SwapObservationAxes,
    RecordEpisodeStatistics,
)
from utils.draw import draw
from utils.timestamp import timestamp


N_ENVS = 1
N_EPISODES = 10000  # Define the total number of episodes to train for
SAVING_INTERVAL = 500  # Save every 500 episodes

ENV_OPTIONS = EnvOptions(
    action_range=1,
    max_steps=800,
    remain_still_threshold=100,
    rewarding=EnvOptions.Rewarding(
        original=0,
        win=50,
        lose=-50,
        ally_defeated=0,
        enemy_defeated=0,
        ally_danger=-200,
        enemy_danger=200,
        invalid_action=-0.25,
        remain_still=0.00,
    ),
    game=GameOptions(
        agent_control=True,
        team_size=1,
        map_size=15,
        gen_trees=True,
        gen_ruins=True,
        random_seed=4048,
        can_flee=False,
        actively_attack=True,
        interval=0.5,
        speed=4,
    ),
)


def main():
    ports = [np.random.randint(10000, 20000) for _ in range(N_ENVS)]
    env = gym.make(
        rimworld_env, options=ENV_OPTIONS, port=ports[0], render_mode="headless"
    )

    env = FrameStackObservation(env, stack_size=8)
    env = SwapObservationAxes(env, swap=(0, 1))
    env = RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    register_keyboard_interrupt(env)
    agent = Agent(
        n_envs=N_ENVS,
        obs_space=env.observation_space,
        act_space=env.action_space[0],
        device="cuda:0",
    )
    agent.policy_net.train()

    next_state, _ = env.reset()

    episode_count = 0  # Initialize episode counter
    with tqdm(total=N_EPISODES, desc="Training Progress (Episodes)") as pbar:
        while episode_count < N_EPISODES:
            current_state = next_state
            actions = agent.act([current_state])

            action = {
                0: actions[0],
            }

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember([current_state], [next_state], [action[0]], [reward], [done])

            agent.train()

            # Calculate completed episodes while preventing overflow
            if done:
                episode_count += 1
                next_state, _ = env.reset()
                # Update episode count and progress bar
                pbar.update(1)

            # Save model and plots at the specified interval
            if (episode_count % SAVING_INTERVAL) <= (N_ENVS / 10) and episode_count > 0:
                agent.policy_net.save(
                    f"agents/dqn/models/{timestamp}/{episode_count:04d}.pth"
                )
                agent.draw_model(
                    f"agents/dqn/plots/training/{timestamp}/{episode_count:04d}.png"
                )
                agent.draw_agent(
                    f"agents/dqn/plots/threshold/{timestamp}/{episode_count:04d}.png"
                )
                draw(
                    env,
                    save_path=f"agents/dqn/plots/env/{timestamp}/{episode_count:04d}.png",
                )
                saving(env, agent, timestamp, episode_count)

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

    os.makedirs(f"agents/dqn/histories/{timestamp}/env/", exist_ok=True)
    os.makedirs(f"agents/dqn/histories/{timestamp}/training/", exist_ok=True)
    os.makedirs(f"agents/dqn/histories/{timestamp}/threshold/", exist_ok=True)

    eps_hist_df.to_csv(
        f"agents/dqn/histories/{timestamp}/env/{episode:04d}.csv",
        index=False,
    )
    stats_df.to_csv(
        f"agents/dqn/histories/{timestamp}/training/{episode:04d}.csv", index=False
    )
    thres_df.to_csv(
        f"agents/dqn/histories/{timestamp}/threshold/{episode:04d}.csv", index=False
    )


if __name__ == "__main__":
    main()

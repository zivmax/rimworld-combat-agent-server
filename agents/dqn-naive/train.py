import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from .agent import DQNAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers import (
    FrameStackObservation,
    SwapObservationAxes,
    RecordEpisodeStatistics,
)
from utils.draw import draw
from utils.timestamp import timestamp


N_ENVS = 1
N_STEPS = int(40e4)  # Total number of steps to train for
SNAPSHOTS = 5

SAVING_INTERVAL = int(N_STEPS / SNAPSHOTS)  # Save every N steps

ENV_OPTIONS = EnvOptions(
    action_range=1,
    max_steps=300,
    rewarding=EnvOptions.Rewarding(
        original=0,
        win=50,
        lose=-50,
        ally_defeated=0,
        enemy_defeated=0,
        ally_danger=-200,
        enemy_danger=200,
        invalid_action=-0.25,
    ),
    game=GameOptions(
        agent_control=True,
        team_size=1,
        map_size=15,
        gen_trees=True,
        gen_ruins=True,
        random_seed=4048,
        can_flee=False,
        actively_attack=False,
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
    env = RecordEpisodeStatistics(env, buffer_length=N_STEPS)
    register_keyboard_interrupt(env)
    agent = Agent(
        n_envs=N_ENVS,
        obs_space=env.observation_space,
        act_space=env.action_space[0],
        device="cuda:0",
    )
    agent.policy_net.train()

    next_state, _ = env.reset()

    step_count = 0  # Initialize step counter
    with tqdm(total=N_STEPS, desc="Training (Steps)") as pbar:
        while step_count < N_STEPS:
            current_state = next_state
            actions = agent.act([current_state])

            action = {
                0: actions[0],
            }

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember([current_state], [next_state], [action[0]], [reward], [done])

            agent.train()

            # Update step count and progress bar
            step_count += 1
            pbar.update(1)

            if done:
                next_state, _ = env.reset()

            # Save model and plots at the specified interval
            if step_count % SAVING_INTERVAL == 0 and step_count > 0:
                agent.policy_net.save(f"agents/dqn/models/{timestamp}/{step_count}.pth")
                agent.draw_model(
                    f"agents/dqn/plots/training/{timestamp}/{step_count}.png"
                )
                agent.draw_agent(
                    f"agents/dqn/plots/threshold/{timestamp}/{step_count}.png"
                )
                draw(
                    env,
                    save_path=f"agents/dqn/plots/env/{timestamp}/{step_count}.png",
                )
                saving(env, agent, timestamp, step_count)

    env.close()


def saving(
    env: RecordEpisodeStatistics, agent: Agent, timestamp: str, steps: int
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
        f"agents/dqn/histories/{timestamp}/env/{steps}.csv",
        index=False,
    )
    stats_df.to_csv(
        f"agents/dqn/histories/{timestamp}/training/{steps}.csv", index=False
    )
    thres_df.to_csv(
        f"agents/dqn/histories/{timestamp}/threshold/{steps}.csv", index=False
    )


from viztracer import VizTracer

tracer = VizTracer(ignore_c_function=True, ignore_frozen=True)
tracer.start()
if __name__ == "__main__":
    try:
        main()
    finally:
        tracer.stop()
        tracer.save(f"agents/dqn/tracings/{timestamp}.json")

import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from agents.dqn import DQNAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers.vector import (
    FrameStackObservation,
    SwapObservationAxes,
    RecordEpisodeStatistics,
)
from utils.draw import draw
from utils.timestamp import timestamp

envs: gym.vector.AsyncVectorEnv = None

N_ENVS = 10
N_STEPS = int(200e4)
SNAPSHOTS = 20

SAVING_INTERVAL = (N_STEPS // SNAPSHOTS) // N_ENVS * N_ENVS

TRAIN_INTERVAL = 100

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
    global envs
    ports = [np.random.randint(10000, 20000) for _ in range(N_ENVS)]
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda port=port: gym.make(rimworld_env, options=ENV_OPTIONS, port=port)
            for port in ports
        ],
        daemon=True,
        shared_memory=True,
    )

    envs = FrameStackObservation(envs, stack_size=8)
    envs = SwapObservationAxes(envs, swap=(0, 1))
    envs = RecordEpisodeStatistics(envs, buffer_length=N_STEPS)
    register_keyboard_interrupt(envs)
    agent = Agent(
        n_envs=N_ENVS,
        obs_space=envs.single_observation_space,
        act_space=envs.single_action_space[0],
        device="cuda",
    )
    agent.policy_net.train()

    next_states, _ = envs.reset()

    steps = 0  # Initialize step counter
    with tqdm(total=N_STEPS, desc="Training Progress (Steps)") as pbar:
        while steps < N_STEPS:
            current_states = next_states
            actions = agent.act(current_states)

            actions = {
                0: [actions[i] for i in range(N_ENVS)],
            }

            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            dones: np.typing.NDArray = np.logical_or(terminateds, truncateds)

            agent.remember(current_states, next_states, actions[0], rewards, dones)

            if steps % TRAIN_INTERVAL == 0 and steps > 0:
                agent.train()

            # Update step count and progress bar
            steps += N_ENVS
            pbar.update(N_ENVS)

            # Save model and plots at the specified interval
            if (steps % SAVING_INTERVAL == 0 and steps > 0) or steps >= N_STEPS:
                agent.policy_net.save(f"agents/dqn/models/{timestamp}/{steps}.pth")
                agent.draw_model(f"agents/dqn/plots/training/{timestamp}/{steps}.png")
                agent.draw_agent(f"agents/dqn/plots/threshold/{timestamp}/{steps}.png")
                draw(
                    envs,
                    save_path=f"agents/dqn/plots/env/{timestamp}/{steps}.png",
                )
                saving(envs, agent, timestamp, steps)

    envs.close()


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
        envs.close()
        tracer.stop()
        tracer.save(f"agents/dqn/tracings/{timestamp}.json")

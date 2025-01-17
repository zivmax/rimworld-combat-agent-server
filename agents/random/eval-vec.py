import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .agent import RandomAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers.vector import (
    FrameStackObservation,
    SwapObservationAxes,
    RecordEpisodeStatistics,
)
from utils.draw import draw
from utils.timestamp import timestamp

envs: gym.vector.AsyncVectorEnv = None

N_ENVS = 50
N_STEPS = int(10e4)

ENV_OPTIONS = EnvOptions(
    action_range=1,
    max_steps=300,
    optimal_range=5,
    range_tolerance=1,
    rewarding=EnvOptions.Rewarding(
        original=0,
        win=50,
        lose=-50,
        ally_defeated=0,
        enemy_defeated=0,
        ally_danger=-200,
        enemy_danger=200,
        invalid_action=-0.5,
        too_close=-0.1,
        too_far=-0.1,
        optimal_distance=0.1,
        cover_reward=0.15,
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
    )

    next_states, _ = envs.reset()

    steps = 0  # Initialize step counter
    step_rewards = []
    with tqdm(total=N_STEPS, desc="Training (Steps)") as pbar:
        while steps < N_STEPS:
            current_states = next_states
            actions = agent.act(current_states)
            next_states, rewards, _, _, _ = envs.step(actions)
            step_rewards.extend(list(rewards))

            # Update step count and progress bar
            steps += N_ENVS
            pbar.update(N_ENVS)
    envs.close()

    # Convert step_rewards to a DataFrame
    df = pd.DataFrame(step_rewards, columns=["reward"])

    # Calculate summary statistics
    summary = df.describe()
    print(summary)

    # Plot rewards
    plt.figure(figsize=(10, 6))
    df["reward"].rolling(window=100).mean().plot()
    plt.title("Average Step Rewards")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.savefig("agents/random/reward_analysis.png")


from viztracer import VizTracer

tracer = VizTracer(ignore_c_function=True, ignore_frozen=True)
tracer.start()
if __name__ == "__main__":
    try:
        main()
    finally:
        envs.close()
        tracer.stop()
        tracer.save(f"agents/random/tracings/{timestamp}.json")

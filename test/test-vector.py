import gymnasium as gym
import signal
import sys
from tqdm import tqdm
from env import rimworld_env, GameOptions, EnvOptions
import numpy as np


def handle_keyboard_interrupt(signum, frame):
    print("KeyboardInterrupt received, closing environments...")
    if "envs" in globals():
        envs.close()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_keyboard_interrupt)

N_EPISODES = 500  # Define the total number of episodes to train for
N_ENV = 20  # Define the number of environments to run in parallel
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
        remain_still=0.05,
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

ports = [np.random.randint(10000, 20000) for _ in range(N_ENV)]
envs = gym.vector.AsyncVectorEnv(
    [
        lambda port=port: gym.make(rimworld_env, options=ENV_OPTIONS, port=port)
        for port in ports
    ],
    daemon=True,
    shared_memory=True,
)

next_states, _ = envs.reset(seed=42)
episode_count = 0  # Initialize episode counter
with tqdm(total=N_EPISODES, desc="Test Progress") as pbar:
    while episode_count < N_EPISODES:
        next_states, rewards, terminateds, truncateds, _ = envs.step(
            envs.action_space.sample()
        )
        dones: np.typing.NDArray = np.logical_or(terminateds, truncateds)

        # Calculate completed episodes while preventing overflow
        completed_episodes = min(dones.sum(), N_EPISODES - episode_count)

        # Update episode count and progress bar
        episode_count += completed_episodes
        pbar.update(completed_episodes)

envs.close()

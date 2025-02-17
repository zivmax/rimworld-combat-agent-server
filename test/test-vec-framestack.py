import gymnasium as gym
import signal
import sys
import tqdm
from env import rimworld_env, GameOptions, EnvOptions
from env.wrappers.vector import FrameStackObservation
import numpy as np


def handle_keyboard_interrupt(signum, frame):
    print("KeyboardInterrupt received, closing environments...")
    if "envs" in globals():
        envs.close()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_keyboard_interrupt)

N_ENVS = 10
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

ports = [np.random.randint(10000, 20000) for _ in range(N_ENVS)]
envs = gym.vector.AsyncVectorEnv(
    [
        lambda port=port: gym.make(rimworld_env, options=ENV_OPTIONS, port=port)
        for port in ports
    ],
    daemon=True,
    shared_memory=True,
)
envs = FrameStackObservation(envs, stack_size=4)

observations, infos = envs.reset(seed=42)
for _ in tqdm.trange(1000):
    observations, rewards, terminations, truncations, infos = envs.step(
        envs.action_space.sample()
    )

envs.close()

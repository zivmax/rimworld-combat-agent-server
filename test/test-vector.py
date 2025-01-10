import gymnasium as gym
from env import rimworld_env
import signal
import sys


def handle_keyboard_interrupt(signum, frame):
    print("KeyboardInterrupt received, closing environments...")
    if "envs" in globals():
        envs.close()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_keyboard_interrupt)

try:
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

    envs = gym.vector.AsyncVectorEnv(
        [
            lambda i=i: gym.make(rimworld_env, options=ENV_OPTIONS, port=10000 + i * 50)
            for i in range(10)
        ],
        daemon=False,
        shared_memory=False,
    )

    observations, infos = envs.reset(seed=42)
    observations, rewards, terminations, truncations, infos = envs.step(
        envs.action_space.sample()
    )

    print(observations, rewards, terminations, truncations, infos)

except Exception as e:
    print(f"An error occurred: {e}")
    if "envs" in globals():
        envs.close()

finally:
    if "envs" in globals():
        envs.close()

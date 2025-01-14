import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from agents.ppo import PPOAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers import (
    FrameStackObservation,
    SwapObservationAxes,
    RecordEpisodeStatistics,
)
from utils.draw import draw
from utils.timestamp import timestamp

N_ENVS = 1
<<<<<<< HEAD
N_STEPS = int(40)
=======
N_STEPS = int(2e1)
>>>>>>> 6567439f8ba5d32e95043c12d3c23976c4a2948b
SAVING_INTERVAL = int((N_STEPS / N_ENVS) * 0.2)
UPDATE_INTERVAL = int((N_STEPS / N_ENVS) * 0.05)

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


def create_env():
    return gym.make(rimworld_env, options=ENV_OPTIONS, render_mode="headless")


def main():
    n_steps = int(N_STEPS / N_ENVS)
    ports = [np.random.randint(10000, 20000) for _ in range(N_ENVS)]
    env = gym.make(
        rimworld_env, options=ENV_OPTIONS, port=ports[0], render_mode="headless"
    )

    env = FrameStackObservation(env, stack_size=8)
    env = SwapObservationAxes(env, swap=(0, 1))
    env = RecordEpisodeStatistics(env, buffer_length=n_steps)
    register_keyboard_interrupt(env)
    agent = Agent(
        n_envs=N_ENVS,
        obs_space=env.observation_space,
        act_space=env.action_space[0],
    )

<<<<<<< HEAD
    next_states, _ = envs.reset()
    for step in tqdm(range(1, n_steps + 1), desc="Training Progress"):
        current_states = next_states
        actions = agent.select_action(current_states)
=======
    next_state, _ = env.reset()
    for step in tqdm(range(1, n_steps + 1), desc="Training Progress (Steps)"):
        current_state = next_state
        actions = agent.select_action([current_state])
>>>>>>> 6567439f8ba5d32e95043c12d3c23976c4a2948b

        action = {
            0: actions[0],
        }
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(reward, next_state, done)

        if step % UPDATE_INTERVAL == 0:
            agent.update()

        if step % SAVING_INTERVAL == 0 and step > 0:
            agent.policy.save(f"agents/ppo/models/{timestamp}/{step:04d}.pth")
            draw(
                env,
                save_path=f"agents/ppo/plots/env/{timestamp}/{step:04d}.png",
            )
            saving(env, agent, timestamp, step)

    env.close()


def saving(
    env: RecordEpisodeStatistics, agent: Agent, timestamp: str, episode: int
) -> None:

    eps_hist_df = pd.DataFrame(
        {
            "Episode": range(len(env.return_queue)),
            "Rewards": env.return_queue,
            "Length": env.length_queue,
            "Time": env.time_queue,
        }
    )

    stats_df = pd.DataFrame(
        {
            "Update": range(len(agent.loss_history)),
            "Loss": agent.loss_history,
            "Policy Loss": agent.policy_loss_history,
            "Value Loss": agent.value_loss_history,
        }
    )

    os.makedirs(f"agents/ppo/histories/{timestamp}/env/", exist_ok=True)
    os.makedirs(f"agents/ppo/histories/{timestamp}/training/", exist_ok=True)

    eps_hist_df.to_csv(
        f"agents/ppo/histories/{timestamp}/env/{episode:04d}.csv",
        index=False,
    )
    stats_df.to_csv(
        f"agents/ppo/histories/{timestamp}/training/{episode:04d}.csv", index=False
    )


if __name__ == "__main__":
    from viztracer import VizTracer

    tracer = VizTracer(ignore_c_function=True, ignore_frozen=True)
    tracer.start()
    try:
        main()
    finally:
        tracer.stop()
        tracer.save("agents/ppo/logs/tracing.json")

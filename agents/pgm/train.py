import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from agents.pgm import PGAgent as Agent  # Updated import path
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers import (
    FrameStackObservation,
    SwapObservationAxes,
    RecordEpisodeStatistics,
)
from utils.draw import draw
from utils.timestamp import timestamp

N_ENVS = 1
N_STEPS = int(2e3)
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

    next_state, _ = env.reset()
    for step in tqdm(range(1, n_steps + 1), desc="Training Progress (Steps)"):
        current_state = next_state
        actions, log_probs = agent.act([current_state])

        action = {
            0: actions[0],
        }
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.remember(
            current_state, actions[0], log_probs[0], reward, next_state, done
        )

        if done:
            next_state, _ = env.reset()

        if step % UPDATE_INTERVAL == 0:
            agent.train()

        if step % SAVING_INTERVAL == 0 and step > 0:
            agent.policy.save(f"agents/pg/models/{timestamp}/{step}.pth")
            draw(
                env,
                save_path=f"agents/pg/plots/env/{timestamp}/{step}.png",
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
        }
    )

    os.makedirs(f"agents/pg/histories/{timestamp}/env/", exist_ok=True)
    os.makedirs(f"agents/pg/histories/{timestamp}/training/", exist_ok=True)

    eps_hist_df.to_csv(
        f"agents/pg/histories/{timestamp}/env/{episode}.csv",
        index=False,
    )
    stats_df.to_csv(
        f"agents/pg/histories/{timestamp}/training/{episode}.csv", index=False
    )


from viztracer import VizTracer

tracer = VizTracer(ignore_c_function=True, ignore_frozen=True)
tracer.start()
if __name__ == "__main__":
    try:
        main()
    finally:
        tracer.stop()
        tracer.save("agents/pg/logs/tracing.json")

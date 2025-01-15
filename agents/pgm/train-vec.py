import gymnasium as gym
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from agents.pgm import PGAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers.vector import FrameStackObservation, SwapObservationAxes
from utils.draw import draw
from utils.timestamp import timestamp

envs: AsyncVectorEnv = None

N_ENVS = 5
N_STEPS = int(150e4)
SNAPSHOTS = 20

SAVING_INTERVAL = int(N_STEPS / SNAPSHOTS)
TRAIN_BATCH_SIZE = 2000

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
    global envs
    ports = [np.random.randint(10000, 20000) for _ in range(N_ENVS)]
    envs = AsyncVectorEnv(
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
    steps = 0
    with tqdm(total=N_STEPS, desc="Training Progress") as pbar:
        while steps < N_STEPS:
            current_states = next_states
            raw_actions, log_probs = agent.act(current_states)
            actions = {
                0: [raw_actions[i] for i in range(N_ENVS)],
            }
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(N_ENVS):
                agent.remember(
                    current_states[i],
                    raw_actions[i],
                    log_probs[i],
                    rewards[i],
                    next_states[i],
                    dones[i],
                )

            if steps % TRAIN_BATCH_SIZE == 0:
                agent.train()

            if steps % SAVING_INTERVAL == 0 and steps > 0:
                agent.policy.save(f"agents/ppo/models/{timestamp}/{steps}.pth")
                draw(
                    envs,
                    save_path=f"agents/ppo/plots/env/{timestamp}/{steps}.png",
                )
                saving(envs, agent, timestamp, steps)

            pbar.update(N_ENVS)
            steps += N_ENVS

    envs.close()


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

    os.makedirs(f"agents/ppo/histories/{timestamp}/env/", exist_ok=True)
    os.makedirs(f"agents/ppo/histories/{timestamp}/training/", exist_ok=True)

    eps_hist_df.to_csv(
        f"agents/ppo/histories/{timestamp}/env/{episode:04d}.csv",
        index=False,
    )
    stats_df.to_csv(
        f"agents/ppo/histories/{timestamp}/training/{episode:04d}.csv", index=False
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
        tracer.save(f"agents/ppo/tracing/{timestamp}.json")

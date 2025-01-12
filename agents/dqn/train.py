import gymnasium as gym
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


from agents.dqn import DQNAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers.vector import FrameStackObservation, SwapObservationAxes
from utils.draw import draw
from utils.timestamp import timestamp

N_ENVS = 2
N_STEPS = int(100e4)
SAVING_INTERVAL = 50000

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


def main():
    n_steps = int(N_STEPS / N_ENVS)
    saving_interval = int(SAVING_INTERVAL / N_ENVS)
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
    envs = RecordEpisodeStatistics(envs, buffer_length=n_steps)
    register_keyboard_interrupt(envs)
    agent = Agent(
        n_envs=N_ENVS,
        obs_space=envs.single_observation_space,
        act_space=envs.single_action_space[0],
    )
    agent.policy_net.train()

    next_states, _ = envs.reset()

    for step in tqdm(range(1, n_steps + 1), desc="Training Progress"):
        current_states = next_states
        actions = agent.act(current_states)

        actions = {
            0: [actions[i] for i in range(N_ENVS)],
        }

        next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        agent.remember(current_states, next_states, actions[0], rewards, dones)

        agent.train()

        if step % saving_interval == 0 and step > 0:
            agent.policy_net.save(f"agents/dqn/models/{timestamp}/{step:04d}.pth")
            agent.draw_model(f"agents/dqn/plots/training/{timestamp}/{step:04d}.png")
            agent.draw_agent(f"agents/dqn/plots/threshold/{timestamp}/{step:04d}.png")
            draw(
                envs,
                save_path=f"agents/dqn/plots/env/{timestamp}/{step:04d}.png",
            )
            saving(envs, agent, timestamp, step)

    envs.close()


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

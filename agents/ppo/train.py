# INSERT_YOUR_REWRITE_HERE
import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, FrameStackObservation
from agents.ppo import PPOAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from utils.draw import draw
from utils.timestamp import timestamp
import pandas as pd
import os

ENV_OPTIONS = EnvOptions(
    action_range=1,
    max_steps=800,
    is_remote=False,
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

N_EPISODES = 10000
SAVING_INTERVAL = 200
UPDATE_INTERVAL = 1024

def main():
    env = gym.make(rimworld_env, options=ENV_OPTIONS)
    env = FrameStackObservation(env, stack_size=4)
    env = RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    register_keyboard_interrupt(env)
    agent = Agent(obs_space=env.observation_space, act_space=env.action_space)


    episode_rewards = []
    try:
        for episode in tqdm(range(1, N_EPISODES + 1), desc="Training Progress"):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(
                    reward,
                    next_state,
                    done,
                )
                episode_reward += reward
                state = next_state

                if len(agent.memory.transitions) >= UPDATE_INTERVAL:
                    agent.update()

            episode_rewards.append(episode_reward)

            if episode % SAVING_INTERVAL == 0:
                agent.policy_net.save(f"agents/ppo/models/{timestamp}/ppo_{episode}.pth")
                agent.draw_model(f"agents/ppo/plots/training/{timestamp}/ppo_{episode}.png")
                agent.draw_agent(f"agents/ppo/plots/threshold/{timestamp}/ppo_{episode}.png")
                draw(env, save_path=f"agents/ppo/plots/env/{timestamp}/env_{episode}.png")
                saving(env, agent, timestamp, episode)

    except Exception as e:
        print(f"An error occurred: {e}")
        env.close()

    finally:
        env.close()

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
            "Policy Loss": agent.policy_loss_history,
            "Value Loss": agent.value_loss_history,
            # Add more metrics if available
        }
    )

    # Create a DataFrame with the threshold history
    thres_df = pd.DataFrame(
        {
            "Steps": range(len(agent.eps_threshold_history)),
            "Threshold": agent.eps_threshold_history,
        }
    )

    os.makedirs(f"agents/ppo/histories/{timestamp}/env/", exist_ok=True)
    os.makedirs(f"agents/ppo/histories/{timestamp}/training/", exist_ok=True)
    os.makedirs(f"agents/ppo/histories/{timestamp}/threshold/", exist_ok=True)

    eps_hist_df.to_csv(
        f"agents/ppo/histories/{timestamp}/env/{episode:04d}.csv",
        index=False,
    )
    stats_df.to_csv(
        f"agents/ppo/histories/{timestamp}/training/{episode:04d}.csv", index=False
    )
    thres_df.to_csv(
        f"agents/ppo/histories/{timestamp}/threshold/{episode:04d}.csv", index=False
    )

if __name__ == "__main__":
    main()

import gymnasium as gym
from tqdm import tqdm

from agents.ppo import PPOAgent as Agent
from env import rimworld_env, GameOptions, EnvOptions, register_keyboard_interrupt
from env.wrappers import (
    FrameStackObservation,
    SwapObservationAxes,
)

N_EPISODES = int(5)  # Total number of steps to train for
MODEL = "agents/dqn/models/2025-01-17_01:47:30/200000.pth"


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
        speed=1,
    ),
)


def main():
    env = gym.make(
        rimworld_env,
        options=ENV_OPTIONS,
        port=ENV_OPTIONS.game.server_port,
        render_mode="human",
    )
    env = FrameStackObservation(env, stack_size=8)
    env = SwapObservationAxes(env, swap=(0, 1))
    register_keyboard_interrupt(env)
    agent = Agent(
        n_envs=1,
        obs_space=env.observation_space,
        act_space=env.action_space[0],
        device="cuda:1",
    )
    agent.policy.load(MODEL)
    agent.policy.eval()

    next_state, _ = env.reset()

    done = False
    rewards = [0] * N_EPISODES
    with tqdm(total=N_EPISODES, desc="Testing (Episodes)") as pbar:
        for i in range(N_EPISODES):
            while not done:
                current_state = next_state
                actions, _ = agent.act([current_state])

                action = {
                    0: actions[0],
                }

                next_state, reward, terminated, truncated, _ = env.step(action)
                rewards[i] += reward
                done = terminated or truncated
                if done:
                    break
            done = False
            next_state, _ = env.reset()
            pbar.update(1)
    env.close()
    print(rewards)


if __name__ == "__main__":
    main()

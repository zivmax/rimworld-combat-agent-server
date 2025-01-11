import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, FrameStackObservation
from agents.ppo import PPOAgent as Agent
from env import rimworld_env
from utils.draw import draw
from utils.timestamp import timestamp

ENV_OPTIONS = {
    "interval": 0.5,
    "speed": 3,
    "action_range": 1,
    "max_steps": 800,
    "is_remote": False,
    "remain_still_threshold": 100,
    "rewarding": {
        "original": 0,
        "win": 0,
        "lose": -0,
        "ally_defeated": -100,
        "enemy_defeated": 100,
        "ally_danger": -200,
        "enemy_danger": 200,
        "invalid_action": -0.25,
        "remain_still": -0.25,
    },
}


N_EPISODES = 10000
SAVING_INTERVAL = 200
UPDATE_INTERVAL = 1024


def main():
    env = gym.make(rimworld_env, options=ENV_OPTIONS)
    env = FrameStackObservation(env, stack_size=4)
    env = RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    agent = Agent(obs_space=env.observation_space, act_space=env.action_space)

    episode_rewards = []
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
            agent.policy.save(f"agents/ppo/models/{timestamp}/ppo_{episode}.pth")
            draw(env, save_path=f"agents/ppo/plots/{timestamp}/env_{episode}.png")

    env.close()


if __name__ == "__main__":
    main()

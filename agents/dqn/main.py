from agents.dqn.dqn import DQNAgent, DQNTrainer


def main(env):
    agent = DQNAgent(
        state_dim=(
            env.observation_space.shape[0],
            env.observation_space.shape[1],
        ),
    )
    trainer = DQNTrainer(env, agent)
    trainer.train()

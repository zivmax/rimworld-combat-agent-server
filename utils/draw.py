import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordEpisodeStatistics


def draw(env: RecordEpisodeStatistics, save_path: str = "./env_history.png") -> None:
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a DataFrame with the episode statistics
    stats_df = pd.DataFrame(
        {
            "Episode": range(len(env.return_queue)),
            "Rewards": env.return_queue,
            "Length": env.length_queue,
            "Time": env.time_queue,
        }
    )

    # Create subplots for each metric
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot rewards
    sns.lineplot(data=stats_df, x="Episode", y="Rewards", ax=ax1)
    ax1.set_title("Episode Rewards over Time")

    # Plot episode lengths
    sns.lineplot(data=stats_df, x="Episode", y="Length", ax=ax2)
    ax2.set_title("Episode Lengths over Time")

    # Plot time taken
    sns.lineplot(data=stats_df, x="Episode", y="Time", ax=ax3)
    ax3.set_title("Episode Time Taken")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

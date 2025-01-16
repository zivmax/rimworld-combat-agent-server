import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Optional
import os


def calculate_rolling_average(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Calculate the rolling average of the 'Rewards' column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window_size (int): The size of the rolling window.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'Rolling_Avg_Rewards' column.
    """
    df["Rolling_Avg_Rewards"] = (
        df["Rewards"].rolling(window=window_size, min_periods=1).mean()
    )
    return df


def plot_rewards(df: pd.DataFrame, save_path: str, window_size: int = 10) -> None:
    """
    Plot the original rewards and the rolling average as line plots.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window_size (int): The size of the rolling window.
    """
    df["Rewards_MA"] = df["Rewards"].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df["Episode"], df["Rewards"], label="Rewards", alpha=0.5)
    plt.plot(
        df["Episode"],
        df["Rewards_MA"],
        label=f"Rewards (MA, window={window_size})",
        color="red",
    )
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Rewards with Moving Average")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)


def main(file_path: str, window_size: Optional[int] = 10) -> None:
    """
    Main function to read the CSV file, calculate the rolling average, and plot the results.

    Args:
        file_path (str): Path to the CSV file.
        window_size (Optional[int]): Size of the rolling window. Defaults to 5.
    """
    # Load the CSV data into a DataFrame
    df: pd.DataFrame = pd.read_csv(file_path)

    # Calculate the rolling average
    df = calculate_rolling_average(df, window_size)

    # Determine the save path for the plot
    save_dir = os.path.dirname(file_path)
    save_path = os.path.join(save_dir, "timeshift.png")

    # Plot the results
    plot_rewards(df, save_path, window_size)


if __name__ == "__main__":
    # Set up argument parsing
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Plot a time-shifted average graph from a CSV file."
    )
    parser.add_argument("file_path", type=str, help="Path to the CSV file.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Size of the rolling window (default: 5).",
    )

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # Run the main function
    main(args.file_path, args.window_size)

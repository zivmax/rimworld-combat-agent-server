from env.game import Game, GameOptions
from time import sleep
from tqdm import tqdm

# Path to the RimWorld executable
rimworld_path = "/mnt/game/RimWorldLinux"

# Number of restarts
RESTART_COUNT = 5

options = GameOptions(
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
)


try:
    # Create progress bar
    with tqdm(total=RESTART_COUNT, desc="Game Tests") as pbar:
        game = Game(rimworld_path, options)

        for i in range(RESTART_COUNT):
            pbar.set_description(f"Running test {i + 1}/{RESTART_COUNT}")
            try:

                # Create an instance of the launcher with custom server address and port
                # Launch the game
                game.launch()

                # Wait for some time to let the game run
                sleep(60)

            except Exception as e:
                print(f"Error during iteration {i + 1}: {e}")

            finally:
                # Make sure to shut down the game properly
                game.restart()
                sleep(2)  # Give some time between restarts
                pbar.update(1)

except KeyboardInterrupt:
    print("\nTest interrupted by user")
    if "game" in locals():
        game.shutdown()

print("Test complete")

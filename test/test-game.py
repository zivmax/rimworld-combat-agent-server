from env.game import Game
import time


# Path to the RimWorld executable
rimworld_path = "/mnt/game/RimWorldLinux"

# Create an instance of the launcher with custom server address and port
game = Game(rimworld_path, server_addr="192.168.1.100", port=12345)

# Launch the game
game.launch()

# Keep the script running to monitor the game process
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Script interrupted by user. Shutting down...")
    game.shutdown()

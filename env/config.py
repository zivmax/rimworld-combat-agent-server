import logging

# Full Reset Interval in episodes
FULL_RESET_INTERVAL = 5

# Restart the game every `RESTART_INTERVAL` episodes
RESTART_INTERVAL = FULL_RESET_INTERVAL * 100  # 0 means never restart

# Timeout for reset in seconds
RESET_TIMEOUT = 30

# Timeout for staring the game
START_TIMEOUT = 300

# Timeout for game response in seconds
RESPONSE_TIMEOUT = 5


RIMWORLD_LOGGING_LEVEL = logging.INFO  # Logging level for RimWorld game process
SERVER_LOGGING_LEVEL = logging.INFO  # Logging level for the server process
STATE_COLLECTOR_LOGGING_LEVEL = (
    logging.INFO
)  # Logging level for the state collector process

import logging

# Restart the game every `RESTART_INTERVAL` episodes
RESTART_INTERVAL = 0  # 0 means never restart

# Timeout for reset in seconds
RESET_TIMEOUT = 30

RIMWORLD_LOGGING_LEVEL = logging.INFO  # Logging level for RimWorld game process
SERVER_LOGGING_LEVEL = logging.INFO  # Logging level for the server process
STATE_COLLECTOR_LOGGING_LEVEL = (
    logging.INFO
)  # Logging level for the state collector process

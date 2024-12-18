import logging
from utils.logger import get_file_logger, get_cli_logger, timestamp

logging_level = logging.DEBUG
f_logger = get_file_logger(f"agents/dqn/logs/{timestamp}.log", logging_level)
cli_logger = get_cli_logger(logging_level)

logger = f_logger

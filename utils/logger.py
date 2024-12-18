import os
import logging

from utils.color import ColoredFormatter


def get_cli_logger(logger_name: str, level) -> logging.Logger:
    """
    Create a logger that prints logs to the console.
    """

    # Configure logging with colored formatter
    cli_handler = logging.StreamHandler()
    cli_handler.setFormatter(
        ColoredFormatter("%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s")
    )
    logger = logging.getLogger("cli-" + logger_name)
    logger.addHandler(cli_handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_file_logger(logger_name: str, path: str, level) -> logging.Logger:
    """
    Create a logger that saves logs to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s")
    )
    logger = logging.getLogger("file-" + logger_name)
    logger.addHandler(file_handler)
    logger.setLevel(level)  # Add this line
    logger.propagate = False
    return logger

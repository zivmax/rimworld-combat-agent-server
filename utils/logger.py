import os
import logging

from utils.color import ColoredFormatter


def get_cli_logger(logger_name: str, level) -> logging.Logger:
    """
    Create a logger that prints logs to the console.
    """

    # Configure logging with colored formatter
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColoredFormatter("%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logger = logging.getLogger("cli-" + logger_name)
    logger.setLevel(level)
    return logger


def get_file_logger(logger_name: str, path: str, level) -> logging.Logger:
    """
    Create a logger that saves logs to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s")
    )
    logger = logging.getLogger("file-" + logger_name)
    logger.addHandler(file_handler)
    return logger

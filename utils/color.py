from colorama import Fore, Style
import colorama
import logging

# Initialize colorama for Windows support
colorama.init()


# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.BLUE,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

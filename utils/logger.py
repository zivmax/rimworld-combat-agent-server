from logging import getLogger
import logging
from utils.color import ColoredFormatter

# Configure logging with colored formatter
handler = logging.StreamHandler()
handler.setFormatter(
    ColoredFormatter("%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s")
)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logger = getLogger(__name__)

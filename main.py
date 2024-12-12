from utils.server import server
from utils.logger import logger
from agent.state import StateCollector
from threading import Thread


def create_server_thread():
    thread = Thread(target=server.start, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    StateCollector.initialize()
    server_thread = create_server_thread()

    try:
        server_thread.join()
    except KeyboardInterrupt:
        logger.info("Shutting down server...\n")
        server.stop()
        exit(0)

from utils.server import server
from utils.logger import logger
from agent.state import StateCollector
from agent.action import random_action_test
from threading import Thread


def create_server_thread():
    thread = Thread(target=server.start, daemon=True)
    thread.start()
    return thread


def create_agent_thread():

    def start_agent():
        while True:
            try:
                StateCollector.collect_state()
                random_action_test()
            except KeyboardInterrupt:
                logger.info("Shutting down agent...\n")
                break

    thread = Thread(target=start_agent(), daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    server_thread = create_server_thread()
    agent_thread = create_agent_thread()

    agent_thread.join()
    logger.info("Agent stopped\n")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        logger.info("Shutting down server...\n")
        server.stop()

    exit(0)

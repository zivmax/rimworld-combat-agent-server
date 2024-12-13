from utils.server import server
from utils.logger import logger
from agent.state import StateCollector, GameStatus
from agent.random import RandomAgent

from threading import Thread, Event
import signal
import sys

# Global flag to indicate if the program should stop
stop_event = Event()


def create_server_thread():
    thread = Thread(target=server.start, daemon=True)
    thread.start()
    return thread


def create_agent_thread():

    def start_agent():
        agent = RandomAgent()
        while True:
            StateCollector.receive_state()
            if StateCollector.current_state.status == GameStatus.RUNNING:
                action = agent.act(StateCollector.current_state)
                message = {
                    "Type": "GameAction",
                    "Data": dict(action),
                }
                server.send_to_client(server.client, message)

                logger.info(
                    f"\tSent actions to clients at tick {StateCollector.current_state.tick}\n"
                )
            else:
                logger.info("\tEpisode Ends.\n")
                message = {"Type": "Reset", "Data": None}
                StateCollector.current_state = None
                server.send_to_client(server.client, message)

    thread = Thread(target=start_agent(), daemon=True)
    thread.start()
    return thread


def signal_handler(sig, frame):
    logger.info("\tStopping threads...\n")
    stop_event.set()
    server.stop()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    server_thread = create_server_thread()
    agent_thread = create_agent_thread()

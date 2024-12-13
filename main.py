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
        while True:
            StateCollector.collect_state()
            if StateCollector.current_state.status == GameStatus.RUNNING:
                agent = RandomAgent()
                action = agent.act()

                message = {
                    "Type": "GameAction",
                    "Data": dict(action),
                }
                for client in server.clients:
                    server.send_to_client(client, message)

                logger.info(
                    f"Sent actions to clients at tick {StateCollector.current_state.tick}\n"
                )

    thread = Thread(target=start_agent(), daemon=True)
    thread.start()
    return thread


def signal_handler(sig, frame):
    logger.info("Stopping threads...\n")
    stop_event.set()
    server.stop()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    server_thread = create_server_thread()
    agent_thread = create_agent_thread()

from utils.server import server
from utils.logger import logger
from agent.state import StateCollector, GameStatus
from agent.random import RandomAgent

from threading import Thread


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


if __name__ == "__main__":
    server_thread = create_server_thread()
    agent_thread = create_agent_thread()

    try:
        agent_thread.join()
        server_thread.join()
    except KeyboardInterrupt:
        logger.info("Agent stopped\n")
        server.stop()
        logger.info("Server stopped\n")

    exit(0)

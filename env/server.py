import socket
import json
import logging
from threading import Thread, Event
from typing import Tuple, Dict, Any
from queue import Queue
from socket import socket as Socket
import signal
import sys

from utils.logger import get_file_logger, get_cli_logger
from utils.timestamp import timestamp

logging_level = logging.DEBUG
f_logger = get_file_logger(__name__, f"env/logs/server/{timestamp}.log", logging_level)
cli_logger = get_cli_logger(__name__, logging_level)
logger = f_logger

stop_event = Event()


def signal_handler(sig, frame, server: "GameServer") -> None:
    logger.info("\nStopping threads...")
    stop_event.set()
    server.stop()
    sys.exit(0)


class GameServer:
    DEFAULT_HOST: str = "localhost"
    REMOTE_HOST: str = "0.0.0.0"
    PORT: int = 10086
    BUFFER_SIZE: int = 5120
    ENCODING: str = "utf-8"
    QUEUE_SIZE: int = 10

    def __init__(self, host: str = DEFAULT_HOST, port: int = PORT) -> None:
        self.host = host
        self.port = port
        self.running: bool = True
        self.client: Socket = None
        self.client_addr: Tuple[str, int] = None
        self.server: Socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.message_queue: Queue = Queue(self.QUEUE_SIZE)

    def start(self, is_remote: bool = False) -> None:
        self.host = self.REMOTE_HOST if is_remote else self.DEFAULT_HOST
        self.server.bind((self.host, self.port))
        self.server.listen(1)

        logger.info(f"Server starting on {self.host}:{self.port}")
        while self.running:
            try:
                self.client, self.client_addr = self.server.accept()
                logger.info(f"Connected to client at {self.client_addr}")

                client_thread = Thread(target=self.handle_client, daemon=True)
                client_thread.start()

            except Exception as e:
                logger.error(f"Error accepting connection: {e}")

    def handle_client(self) -> None:
        buffer = ""
        try:
            while self.running:
                try:
                    data = self.client.recv(self.BUFFER_SIZE)

                    if not data:
                        self.client.close()
                        self.client = None
                        logger.info(f"Client {self.client_addr} disconnected")
                        break

                    buffer += data.decode(self.ENCODING)

                    while "\n" in buffer:
                        message, buffer = buffer.split("\n", 1)
                        try:
                            data = json.loads(message)
                            if data["Type"] == "Log":
                                logger.debug(
                                    f"Client {self.client_addr}: {data['Data']}"
                                )
                                continue
                            self.message_queue.put(data)
                            logger.debug(
                                f"Received data from {self.client_addr}, Type: {data['Type']}"
                            )

                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON from {self.client_addr}: {e}")
                            logger.error(f"Raw String:\n {message}")
                            continue

                except (ConnectionResetError, ConnectionAbortedError) as e:
                    self.client.close()
                    self.client = None
                    logger.info(f"Client {self.client_addr} disconnected: {e}")
                    break
                except Exception as e:
                    logger.error(
                        f"Unexpected error handling client {self.client_addr}: {e}"
                    )
                    break
        finally:
            self.client.close()

    def stop(self) -> None:
        self.running = False
        try:
            if self.client:
                self.client.close()
        except Exception as e:
            logger.error(f"Error closing client connection: {e}")
        self.server.close()
        logger.info("\nServer stopped")

    def send_to_client(self, message: str) -> bool:
        """Send a message to a specific client"""
        try:
            json_string = json.dumps(message)
            self.client.send(f"{json_string}\n".encode(self.ENCODING))
            logger.debug(f"Sent message to {self.client_addr}, Type: {message['Type']}")
            return True
        except (ConnectionResetError, ConnectionAbortedError):
            self.client.close()
            return False

    @classmethod
    def create_server_thread(
        cls, is_remote: bool = False
    ) -> Tuple[Thread, "GameServer"]:
        server = GameServer()
        signal.signal(
            signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, server)
        )
        thread = Thread(target=server.start, daemon=True, args=(is_remote,))
        thread.start()
        return thread, server

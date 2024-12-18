import socket
import json
import logging
from threading import Thread
from typing import Tuple, Callable, Dict, Any
from queue import Queue
from socket import socket as Socket

from utils.logger import get_file_logger, get_cli_logger
from utils.timestamp import timestamp

import signal
import sys
from threading import Event

stop_event = Event()


logging_level = logging.INFO
f_logger = get_file_logger(__name__, f"env/logs/server/{timestamp}.log", logging_level)
cli_logger = get_cli_logger(__name__, logging_level)

logger = cli_logger


def signal_handler(sig, frame) -> None:
    logger.info("\nStopping threads...")
    stop_event.set()
    server.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def create_server_thread(is_remote: bool = False) -> Thread:
    thread = Thread(target=server.start, daemon=True, args=([is_remote]))
    thread.start()
    return thread


class GameServer:
    HOST: str = "localhost"
    PORT: int = 10086
    BUFFER_SIZE: int = 5120
    ENCODING: str = "utf-8"
    QUEUE_SIZE: int = 10

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self.HOST = host
        self.PORT = port
        self.running: bool = True
        self.client: Socket = None
        self.server: Socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.message_queue: Queue = Queue(self.QUEUE_SIZE)

    def start(self, is_remote: bool = False) -> None:
        if is_remote:
            self.HOST = "0.0.0.0"
        self.server.bind((self.HOST, self.PORT))
        self.server.listen(1)

        logger.info(f"Server starting on {self.HOST}:{self.PORT}")
        while self.running:
            try:
                self.client, addr = self.server.accept()
                logger.info(f"Connected to client at {addr}")

                client_thread = Thread(
                    target=self.handle_client, args=(self.client, addr), daemon=True
                )
                client_thread.start()

            except Exception as e:
                logger.error(f"Error accepting connection: {e}")

    def handle_client(self, client: Socket, addr: Tuple[str, int]) -> None:
        buffer = ""
        try:
            while self.running:
                try:
                    data = self.client.recv(self.BUFFER_SIZE)

                    if not data:
                        self.client.close()
                        self.client = None
                        logger.info(f"Client {addr} disconnected")
                        break

                    buffer += data.decode(self.ENCODING)

                    while "\n" in buffer:
                        message, buffer = buffer.split("\n", 1)
                        try:
                            # Parse JSON message
                            data = json.loads(message)
                            if data["Type"] == "Log":
                                logger.debug(f"Client {addr}: {data['Data']}")
                                continue
                            self.message_queue.put(data)
                            logger.debug(
                                f"Received data from {addr}, type: {data['Type']}"
                            )

                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON from {addr}: {e}")
                            logger.error(f"Raw String:\n {message}")
                            continue

                except (ConnectionResetError, ConnectionAbortedError) as e:
                    self.client.close()
                    self.client = None
                    logger.info(f"Client {addr} disconnected: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error handling client {addr}: {e}")
                    break
        finally:
            client.close()

    def stop(self) -> None:
        self.running = False
        # Close all client connections
        try:
            self.client.close()
        except:
            pass
        self.server.close()
        logger.info("\nServer stopped")

    def send_to_client(self, client: Socket, message: str) -> bool:
        """Send a message to a specific client"""
        try:
            json_string = json.dumps(message)
            self.client.send(f"{json_string}\n".encode(self.ENCODING))
            return True
        except (ConnectionResetError, ConnectionAbortedError):
            self.client.close()
            return False


server = GameServer()

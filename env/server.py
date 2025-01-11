import socket
import json
import logging
from threading import Thread, Event
from typing import Tuple, Dict, Any
from queue import Queue
from socket import socket as Socket

from utils.logger import get_file_logger, get_cli_logger
from utils.timestamp import timestamp

logging_level = logging.INFO
f_logger = get_file_logger(__name__, f"env/logs/server/{timestamp}.log", logging_level)
cli_logger = get_cli_logger(__name__, logging_level)
logger = f_logger

stop_event = Event()


class GameServer:
    DEFAULT_HOST: str = "localhost"
    REMOTE_HOST: str = "0.0.0.0"
    BUFFER_SIZE: int = 5120
    ENCODING: str = "utf-8"
    QUEUE_SIZE: int = 10

    def __init__(self, host: str = DEFAULT_HOST, port: int = 10086) -> None:
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
            if self.client:
                self.client.close()

    def stop(self) -> None:
        self.running = False
        try:
            if self.client:
                self.client.close()
        except Exception as e:
            logger.error(f"Error closing client connection: {e}")
        self.server.close()
        logger.info("Server stopped")

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
        cls, is_remote: bool = False, port: int = 10086
    ) -> Tuple[Thread, "GameServer"]:
        server = GameServer(port=port)
        thread = Thread(target=server.start, daemon=True, args=(is_remote,))
        thread.start()
        return thread, server

    @classmethod
    def find_available_port(
        cls, start_port: int = 10086, max_attempts: int = 100
    ) -> int | None:
        """
        Find and return an available local port using a growth algorithm.

        Args:
            start_port (int): The port number to start checking from.
            max_attempts (int): The maximum number of ports to check.
            growth_factor (int): The factor by which the port interval grows after each attempt.

        Returns:
            int: An available port number, or -1 if no available port is found.
        """
        current_port = start_port
        interval = 1  # Start with a small interval
        attempts = 0
        growth_factor = 10
        grow_threshold = 50

        while attempts < max_attempts:
            try:
                # Create a temporary socket to check if the port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
                    temp_socket.bind(("localhost", current_port))
                    # If bind is successful, the port is available
                    return current_port
            except socket.error:
                attempts += 1

                if attempts % grow_threshold == 0:
                    interval *= growth_factor

                # Port is not available, try the next one with an increased interval
                current_port += interval
                continue

        return None  # No available port found

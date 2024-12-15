import socket
import json
from threading import Thread
from typing import Tuple, Callable, Dict, Any
from queue import Queue
from socket import socket as Socket

from utils.logger import logger

import signal
import sys
from threading import Event

stop_event = Event()


def signal_handler(sig, frame):
    logger.info("\tStopping threads...\n")
    stop_event.set()
    server.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def create_server_thread():
    thread = Thread(target=server.start, daemon=True)
    thread.start()
    return thread


class GameServer:
    HOST: str = "localhost"
    PORT: int = 10086
    BUFFER_SIZE: int = 5120
    ENCODING: str = "utf-8"
    QUEUE_SIZE: int = 10

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self.server: Socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.running: bool = True
        self.client: Socket = None
        self.message_queue: Queue = Queue(self.QUEUE_SIZE)
        self.message_handlers: Dict[str, Callable] = {}

    def register_handler(
        self, message_type: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler

    def start(self) -> None:
        logger.info(f"Server starting on {self.HOST}:{self.PORT}\n")
        while self.running:
            try:
                self.client, addr = self.server.accept()
                logger.info(f"Connected to client at {addr}\n")

                client_thread = Thread(
                    target=self.handle_client, args=(self.client, addr), daemon=True
                )
                client_thread.start()

            except Exception as e:
                logger.error(f"Error accepting connection: {e}\n")

    def handle_client(self, client: Socket, addr: Tuple[str, int]) -> None:
        buffer = ""
        try:
            while self.running:
                try:
                    data = self.client.recv(self.BUFFER_SIZE)

                    if not data:
                        self.client.close()
                        self.client = None
                        logger.info(f"Client {addr} disconnected\n")
                        break

                    buffer += data.decode(self.ENCODING)

                    while "\n" in buffer:
                        message, buffer = buffer.split("\n", 1)
                        try:
                            # Parse JSON message
                            data = json.loads(message)
                            if data["Type"] == "Log":
                                logger.debug(f"Client {addr}: {data['Data']}\n")
                                continue
                            self.message_queue.put(data)
                            logger.debug(
                                f"Received data from {addr}, type: {data['Type']}\n"
                            )

                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON from {addr}: {e}\n")
                            logger.error(f"Raw String:\n {message}\n")
                            continue

                except (ConnectionResetError, ConnectionAbortedError) as e:
                    self.client.close()
                    self.client = None
                    logger.info(f"Client {addr} disconnected: {e}\n")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error handling client {addr}: {e}\n")
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
        logger.info("Server stopped\n")

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

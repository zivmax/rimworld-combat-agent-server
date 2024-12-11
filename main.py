import socket
import json
import threading
from typing import Tuple, Optional
from logging import getLogger
from socket import socket as Socket

logger = getLogger(__name__)

class GameServer:
    HOST: str = 'localhost'
    PORT: int = 10086
    BUFFER_SIZE: int = 5120
    ENCODING: str = 'utf-8'

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self.server: Socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.running: bool = True
        self.clients: set[Socket] = set()
        
    def start(self) -> None:
        logger.info(f"Server starting on {self.HOST}:{self.PORT}\n")
        while self.running:
            try:
                client, addr = self.server.accept()
                logger.info(f"Connected to client at {addr}\n")
                
                self.clients.add(client)
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client, addr),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                logger.error(f"Error accepting connection: {e}\n")
                
    def handle_client(self, client: Socket, addr: Tuple[str, int]) -> None:
        port = addr[-1]
        buffer = ""
        try:
            while self.running:
                try:
                    data = client.recv(self.BUFFER_SIZE)
                    
                    if not data:
                        logger.info(f"Client {port} disconnected gracefully\n")
                        break

                    buffer += data.decode(self.ENCODING)
                    
                    while '\n' in buffer:
                        message, buffer = buffer.split('\n', 1)
                        try:
                            # Parse JSON message
                            json_message = json.loads(message)
                            formatted_message = json.dumps(json_message, indent=4)
                            logger.info(f"Received message from {port}:\n{formatted_message}\n")
                            
                            # Send response back
                            response = "Message received"
                            client.send(f"{response}\n".encode(self.ENCODING))
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON from {port}: {e}\n")
                            logger.error(f"Raw String:\n {message}\n")
                            continue
                        
                except (ConnectionResetError, ConnectionAbortedError) as e:
                    logger.info(f"Client {port} disconnected: {e}\n")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error handling client {port}: {e}\n")
                    break
        finally:
            self.clients.remove(client)
            client.close()

    
    def stop(self) -> None:
        self.running = False
        # Close all client connections
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.server.close()
        logger.info("Server stopped\n")

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    server = GameServer()
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...\n")
        server.stop()
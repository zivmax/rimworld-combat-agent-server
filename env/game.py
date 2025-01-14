import subprocess
from time import sleep
import os
import psutil
import threading
from dataclasses import dataclass
from typing import Optional
import logging

from utils.timestamp import timestamp
from utils.logger import get_cli_logger, get_file_logger

logging_level = logging.INFO
f_logger = get_file_logger(__name__, f"env/logs/manager/{timestamp}.log", logging_level)
cli_logger = get_cli_logger(__name__, logging_level)

logger = f_logger


@dataclass
class GameOptions:
    """
    A dataclass to hold the configuration options for the game.

    Attributes:
        agent_control (bool): Whether agents are controlled by the game (default: True).
        team_size (int): The size of the team (default: 1).
        map_size (int): The size of the map (default: 15).
        gen_trees (bool): Whether to generate trees (default: True).
        gen_ruins (bool): Whether to generate ruins (default: True).
        random_seed (Optional[int]): The random seed for the game (default: 4048).
        can_flee (bool): Whether agents can flee (default: False).
        actively_attack (bool): Whether agents actively attack (default: False).
        interval (float): The interval between game updates (default: 1.0).
        speed (int): The speed of the game (default: 1).
        server_addr (str): The server address (default: "localhost").
        server_port (int): The server port (default: 10086).
    """

    agent_control: bool = True
    team_size: int = 1
    map_size: int = 15
    gen_trees: bool = True
    gen_ruins: bool = True
    random_seed: Optional[int] = 4048
    can_flee: bool = False
    actively_attack: bool = False
    interval: float = 1.0
    speed: int = 1
    server_addr: str = "localhost"
    server_port: int = 10086


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


class Game:
    def __init__(self, game_path, options: Optional[GameOptions] = None):
        """
        Initialize the Game class with the path to the game executable and optional configuration.

        :param game_path: Path to the game executable.
        :param options: Optional GameOptions object to configure the game (default: None).
        """
        self.restarted_times: int = 0  # Number of times the game has been restarted
        self.rimworld_path = game_path
        self.options = options if options else GameOptions()
        self.log_dir = f"env/logs/game/{timestamp}"  # Directory for log files
        self._ensure_log_dir_exists()  # Ensure the log directory exists
        self.process: subprocess.Popen = None  # Store the process object
        self.monitor_thread: threading.Thread = (
            None  # Thread to monitor the process status
        )
        self.log_thread: threading.Thread = None  # Thread to log the process output
        self.logging: bool = True  # Flag to control logging
        self.monitoring: bool = True  # Flag to control monitoring

    def launch(self):
        """
        Launch the game in headless mode with the specified configuration.

        :return: The subprocess.Popen object representing the game process, or None if the launch fails.
        """
        self.logging = True
        self.monitoring = True
        self.stdout_log_file = os.path.join(
            self.log_dir, f"{self.options.server_port}-{self.restarted_times}.log"
        )

        command = [
            self.rimworld_path,
            "-batchmode",
            "-nographics",
            "-disable-gpu-skinning",
            "-no-stereo-rendering",
            "-systemallocator",
            "-quicktest",
            "-headless=True",
            f"-server-addr={self.options.server_addr}",
            f"-server-port={self.options.server_port}",
            f"-agent-control={self.options.agent_control}",
            f"-team-size={self.options.team_size}",
            f"-map-size={self.options.map_size}",
            f"-gen-trees={self.options.gen_trees}",
            f"-gen-ruins={self.options.gen_ruins}",
            f"-seed={self.options.random_seed}",
            f"-can-flee={self.options.can_flee}",
            f"-actively-attack={self.options.actively_attack}",
            f"-interval={self.options.interval}",
            f"-speed={self.options.speed}",
        ]

        try:
            # Start the process and redirect stdout and stderr to PIPE
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",  # Specify the encoding
            )

            # Start a thread to log stdout
            self.log_thread = threading.Thread(
                target=self._log_output,
            )
            self.log_thread.daemon = True
            self.log_thread.start()

            # Start a thread to monitor the process status
            self.monitor_thread = threading.Thread(target=self._monitor_process)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            logger.info(
                f"RimWorld launched in headless mode with server address {self.options.server_addr} and port {self.options.server_port}."
            )
            logger.info(f"Game stdout is being logged to: {self.stdout_log_file}")
            logger.info(f"Game process PID: {self.process.pid}")
            return self.process
        except Exception as e:
            logger.error(f"Failed to launch RimWorld: {e}")
            return None

    def shutdown(self):
        """
        Shutdown the game process and all its child processes.
        """
        if self.process:
            try:
                logger.info(
                    f"Terminating RimWorld game process (PID: {self.process.pid})..."
                )

                # Close the thread for log and monitor
                self.logging = False
                self.monitoring = False

                # Terminate the parent process
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    kill(self.process.pid)
                sleep(5)  # Add a short delay after termination
                logger.info("RimWorld game process has been terminated.")
            except Exception as e:
                logger.error(f"Failed to terminate RimWorld game process: {e}")
            finally:
                self.process = None  # Reset the process object
        else:
            logger.warning("Shutting down a not found process.")

    def restart(self):
        """
        Restart the RimWorld game process.
        """
        logger.info("Restarting RimWorld...")
        self.shutdown()  # Shutdown the current process
        self.restarted_times += 1

        # Close the old log file if it's open
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()

        self.launch()  # Launch the game again

    def _ensure_log_dir_exists(self):
        """
        Ensure the log directory exists. If not, create it.
        """
        log_path = os.path.join(os.getcwd(), self.log_dir)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    def _monitor_process(self):
        """
        Monitor the process status and restart the game if it crashes.
        """
        while True:
            if not self.monitoring:
                break
            if self.process and self.process.poll() is not None:
                pid = self.process.pid
                returncode = self.process.returncode
                self.process = None  # Reset the process object
                logger.error(
                    f"Game process (PID: {pid}) terminated unexpectedly with return code {returncode}. Restarting..."
                )
                try:
                    self.restart()  # Restart the game
                except Exception as e:
                    logger.error(f"Failed to restart the game: {e}")
                    break  # Exit the monitoring loop if restart fails
            sleep(5)  # Check every 5 seconds

    def _log_output(self):
        """
        Log the output of the process to a file.
        """
        self.log_file = open(self.stdout_log_file, "w")  # Open the log file
        while True:
            if not self.logging:
                break

            output = self.process.stdout.readline()
            if not self.process or (output == "" and self.process.poll() is not None):
                break
            if output:
                self.log_file.write(output)
                self.log_file.flush()
        self.log_file.close()  # Close the log file when done

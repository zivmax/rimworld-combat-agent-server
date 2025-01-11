import subprocess
import time
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


class Game:
    def __init__(self, game_path, options: Optional[GameOptions] = None):
        """
        Initialize the RimWorldHeadlessLauncher with the path to the RimWorld executable,
        server address, and port.

        :param game_path: Path to the RimWorld executable.
        :param server_addr: Server address (default: 127.0.0.1).
        :param port: Port number (default: 10086).
        """
        self.rimworld_path = game_path
        self.options = options if options else GameOptions()
        self.process: subprocess.Popen = None  # Store the process object
        self.log_dir = "env/logs/game"  # Directory for log files
        self.ensure_log_dir_exists()  # Ensure the log directory exists
        self.monitor_thread = None  # Thread to monitor the process status

    def ensure_log_dir_exists(self):
        """
        Ensure the log directory exists. If not, create it.
        """
        log_path = os.path.join(os.getcwd(), self.log_dir)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    def monitor_process(self):
        """
        Monitor the process status and log an error if it terminates unexpectedly.
        """
        while self.process and self.process.poll() is None:
            time.sleep(5)  # Check every 5 seconds

        if self.process and self.process.poll() is not None:
            pid = self.process.pid
            returncode = self.process.returncode
            self.process = None  # Reset the process object
            logger.error(
                f"Game process (PID: {pid}) terminated unexpectedly with return code {returncode}"
            )
            raise Exception(f"Game process (PID: {pid}) terminated unexpectedly")

    def launch(self):
        self.stdout_log_file = os.path.join(self.log_dir, f"{timestamp}.log")

        env_copy = os.environ.copy()
        command = [
            self.rimworld_path,
            "-batchmode",
            "-nographics",
            "-disable-gpu-skinning",
            "-no-stereo-rendering",
            "-systemallocator",
            "-quicktest",
            f"-server={self.options.server_addr}",
            f"-port={self.options.server_port}",
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
                f"{' '.join(command)}",
                # command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env_copy,
                cwd=os.path.dirname(self.rimworld_path),
                text=True,
                bufsize=0,  # Disable buffering
                encoding="utf-8",  # Specify the encoding
                shell=True,
            )

            # Start a thread to log stdout
            log_thread = threading.Thread(
                target=self._log_output,
            )
            log_thread.daemon = True
            log_thread.start()

            # Start a thread to monitor the process status
            self.monitor_thread = threading.Thread(target=self.monitor_process)
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
        if self.process:
            try:
                logger.info(
                    f"Terminating RimWorld game process (PID: {self.process.pid})..."
                )

                # Get the process and all its children
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)

                # Terminate all child processes
                for child in children:
                    child.terminate()

                # Wait for child processes to terminate
                gone, still_alive = psutil.wait_procs(children, timeout=10)
                for child in still_alive:
                    child.kill()  # Forcefully kill any remaining child processes

                # Terminate the parent process
                self.process.terminate()
                self.process.wait(timeout=60)  # Wait for the process to terminate
                logger.info("RimWorld game process has been terminated.")
            except subprocess.TimeoutExpired:
                logger.warning(
                    "RimWorld game process did not terminate gracefully. Forcefully killing..."
                )
                self.process.kill()
                logger.info("RimWorld game process was forcefully terminated.")
            except Exception as e:
                logger.error(f"Failed to terminate RimWorld game process: {e}")
            finally:
                self.process = None  # Reset the process object
        else:
            logger.info("No RimWorld game process is currently running.")

    def restart(self):
        """
        Restart the RimWorld game process.
        """
        logger.info("Restarting RimWorld...")
        self.shutdown()  # Shutdown the current process
        time.sleep(10)  # Add a short delay before relaunching
        self.launch()  # Launch the game again

    def _log_output(self):
        """
        Log the output of the process to a file.

        :param process: The process object.
        :param log_file: The log file to write to.
        :param stream: The stream to read from (stdout or stderr).
        """
        with open(self.stdout_log_file, "w") as log:
            while True:
                output = self.process.stdout.readline()
                if output == "" and self.process.poll() is not None:
                    break
                if output:
                    log.write(output)
                    log.flush()

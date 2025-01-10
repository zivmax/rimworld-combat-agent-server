import subprocess
import time
import os
import psutil
import threading
import logging
from utils.timestamp import timestamp
from utils.logger import get_cli_logger, get_file_logger

logging_level = logging.INFO
f_logger = get_file_logger(__name__, f"env/logs/manager/{timestamp}.log", logging_level)
cli_logger = get_cli_logger(__name__, logging_level)

logger = f_logger


def log_output(process, log_file, stream):
    """
    Log the output of the process to a file.

    :param process: The process object.
    :param log_file: The log file to write to.
    :param stream: The stream to read from (stdout or stderr).
    """
    with open(log_file, "w") as log:
        while True:
            output = stream.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                log.write(output)
                log.flush()


class Game:
    def __init__(self, game_path, server_addr="127.0.0.1", port=10086):
        """
        Initialize the RimWorldHeadlessLauncher with the path to the RimWorld executable,
        server address, and port.

        :param game_path: Path to the RimWorld executable.
        :param server_addr: Server address (default: 127.0.0.1).
        :param port: Port number (default: 10086).
        """
        self.rimworld_path = game_path
        self.server_addr = server_addr
        self.port = port
        self.process = None  # Store the process object
        self.log_dir = "env/logs/game"  # Directory for log files
        self.ensure_log_dir_exists()  # Ensure the log directory exists
        self.monitor_thread = None  # Thread to monitor the process status

    def ensure_log_dir_exists(self):
        """
        Ensure the log directory exists. If not, create it.
        """
        for subdir in ["stdout", "stderr"]:
            log_path = os.path.join(self.log_dir, subdir)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

    def monitor_process(self):
        """
        Monitor the process status and log an error if it terminates unexpectedly.
        """
        while self.process and self.process.poll() is None:
            time.sleep(5)  # Check every 5 seconds

        if self.process and self.process.poll() is not None:
            logger.error(
                f"Game process terminated unexpectedly with return code {self.process.returncode}"
            )
            self.process = None  # Reset the process object

    def launch(self):
        """
        Launch RimWorld in headless mode with the specified server address and port.
        Redirect stdout and stderr to separate log files without blocking.
        """
        # Generate log file names using the timestamp
        stdout_log_file = os.path.join(self.log_dir, f"stdout/{timestamp}.log")
        stderr_log_file = os.path.join(self.log_dir, f"stderr/{timestamp}.log")

        command = [
            self.rimworld_path,
            "-batchmode",
            "-nographics",
            "-quicktest",
            f"-server={self.server_addr}",
            f"-port={self.port}",
        ]

        try:
            # Start the process and redirect stdout and stderr to PIPE
            self.process = subprocess.Popen(
                # f"{' '.join(command)}",
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Disable buffering
                encoding="utf-8",  # Specify the encoding
                shell=False,
            )

            # Start a thread to log stdout
            stdout_thread = threading.Thread(
                target=log_output,
                args=(self.process, stdout_log_file, self.process.stdout),
            )
            stdout_thread.daemon = True
            stdout_thread.start()

            # Start a thread to log stderr
            stderr_thread = threading.Thread(
                target=log_output,
                args=(self.process, stderr_log_file, self.process.stderr),
            )
            stderr_thread.daemon = True
            stderr_thread.start()

            # Start a thread to monitor the process status
            self.monitor_thread = threading.Thread(target=self.monitor_process)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            logger.info(
                f"RimWorld launched in headless mode with server address {self.server_addr} and port {self.port}."
            )
            logger.info(f"Game stdout is being logged to: {stdout_log_file}")
            logger.info(f"Game stderr is being logged to: {stderr_log_file}")
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
                gone, still_alive = psutil.wait_procs(children, timeout=5)
                for child in still_alive:
                    child.kill()  # Forcefully kill any remaining child processes

                # Terminate the parent process
                self.process.terminate()
                self.process.wait(timeout=10)  # Wait for the process to terminate
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

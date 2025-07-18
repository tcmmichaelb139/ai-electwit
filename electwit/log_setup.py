import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = None


def setup_logging():
    """
    Sets up the logging for the entire run.
    - Creates a timestamped directory.
    - Configures a file handler to log to 'electwit.log' inside that directory.
    - Returns the path to the created directory.
    """

    global LOG_DIR

    if LOG_DIR is not None:
        return LOG_DIR

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = Path("logs") / time_str
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file_path = LOG_DIR / "electwit.log"
    log_format = (
        "%(asctime)s [%(levelname)8s] [%(name)s] %(message)s (%(filename)s:%(lineno)s)"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Logs will be saved to {LOG_DIR}")

    return LOG_DIR


def get_log_dir() -> Path:
    """
    Returns the directory where logs are stored.
    If logging has not been set up, it raises an error.
    """
    if LOG_DIR is None:
        raise RuntimeError("Logging has not been set up. Call setup_logging() first.")

    return LOG_DIR

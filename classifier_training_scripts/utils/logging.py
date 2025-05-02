import logging
import os


def setup_logger(output_dir: str,
                  log_file_name: str = 'training.log',
                  level: int = logging.INFO) -> None:
    """
    Initializes the root logger to output messages to both the console and a file.

    Args:
        output_dir (str): Directory where the log file will be saved.
        log_file_name (str): Name of the log file.
        level (int): Logging level (e.g., logging.INFO).
    """
    # Clear existing handlers
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # File handler
    file_path = os.path.join(output_dir, log_file_name)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Logging to console and {file_path}")

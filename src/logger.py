# src/logger.py
import logging
import logging.config
import os
import yaml

DEFAULT_LOGGER_NAME = "fraud"


def setup_logging(
    default_path: str = "config/logging.yaml",
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG",
) -> None:
    """
    Setup logging configuration from YAML file.

    If the config file is missing, falls back to basicConfig.
    Also ensures the logs directory exists if a FileHandler is configured.
    """
    path = os.getenv(env_key, default_path)
    if os.path.exists(path):
        with open(path, "r") as f:
            config = yaml.safe_load(f.read())

        # Ensure log directory exists if a FileHandler is used
        file_handler = config.get("handlers", {}).get("file")
        if file_handler:
            log_file = file_handler.get("filename")
            if log_file:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)
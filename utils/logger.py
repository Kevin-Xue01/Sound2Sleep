import logging
import os

from .config import SessionConfig


class Logger:
    log_format = logging.Formatter(
        '%(created)f <%(name)s> [%(levelname)s] %(message)s'
    )
    def __init__(self, config: SessionConfig, logger_name: str):
        self.config = config
        self.log_file_path = os.path.join(self.config._data_dir, "log.txt")
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)  # Capture all logs; handlers will filter them.
        self.logger.propagate = False  # Prevent logs from propagating to the root logger.

        # Console handler (warnings and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.console_logging_level)
        console_handler.setFormatter(self.log_format)
        self.logger.addHandler(console_handler)

        # File handler (debug and above)
        file_handler = logging.FileHandler(self.log_file_path, mode='a')
        file_handler.setLevel(self.config.file_logging_level)
        file_handler.setFormatter(self.log_format)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Logs a debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Logs an info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Logs a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Logs an error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Logs a critical message."""
        self.logger.critical(message)
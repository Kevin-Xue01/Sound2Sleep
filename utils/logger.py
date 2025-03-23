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
        self.logger.setLevel(logging.DEBUG)

        # Create a console handler for warnings and above
        console_handler = logging.StreamHandler()
        
        console_handler.setFormatter(self.log_format)
        self.logger.addHandler(console_handler)

        self.file_handler = None
        self._setup_file_handler()

    def _setup_file_handler(self):
        self.file_handler = logging.FileHandler(self.log_file_path, mode='a')
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(self.log_format)
        self.logger.addHandler(self.file_handler)
    
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
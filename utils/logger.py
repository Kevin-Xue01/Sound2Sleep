import logging
import os


class Logger:
    log_format = logging.Formatter(
        '%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    def is_valid_path(self, path: str) -> bool:
        try:
            return os.path.exists(path) or os.access(os.path.dirname(path), os.W_OK)
        except Exception:
            return False
    
    def __init__(self, session_key: str, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.session_key = session_key

        # Create a console handler for warnings and above
        console_handler = logging.StreamHandler()
        
        console_handler.setFormatter(self.log_format)
        self.logger.addHandler(console_handler)

        self.file_handler = None
        self._setup_file_handler()

    def _setup_file_handler(self):
        if self.session_key:
            session_log_file_path = os.path.join("data", self.session_key, "log.txt")
            self.file_handler = logging.FileHandler(session_log_file_path, mode='a')
            self.file_handler.setLevel(logging.INFO)
            self.file_handler.setFormatter(self.log_format)
            self.logger.addHandler(self.file_handler)
    
    def update_session_key(self, new_session_key: str):
        """Update session key and reconfigure the logger."""
        if self.session_key != new_session_key:
            self.session_key = new_session_key  # Update session key
            if self.file_handler:
                self.logger.removeHandler(self.file_handler)
                self.file_handler.close()
                self.file_handler = None  # Reset file handler
            self._setup_file_handler()

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
import logging


class Logger:
    def __init__(self, filename: str = "", logger_name: str = "BlueMuse", level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        # Create a console handler for warnings and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        log_format = logging.Formatter(
            '%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(log_format)
        self.logger.addHandler(console_handler)

        # Optionally create a file handler
        if filename != "":
            file_handler = logging.FileHandler(filename, mode='a')
            file_handler.setLevel(level)
            file_handler.setFormatter(log_format)
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
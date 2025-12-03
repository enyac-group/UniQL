import logging
import sys
import os

try:
    import colorama
    colorama.init()
except ImportError:
    colorama = None

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[0m',       # Default
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def supports_ansi_color():
    if os.name == 'nt' and colorama is None:
        return False
    return sys.stdout.isatty()


def set_logger(logger, level=logging.INFO):
    logger.setLevel(level)
    
    # Clear any existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Also clear handlers from parent loggers to prevent propagation
    logger.propagate = False

    handler = logging.StreamHandler()
    if supports_ansi_color():
        formatter = ColorFormatter(
            fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S"
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
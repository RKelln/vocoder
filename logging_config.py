# logging_config.py
import logging
import sys


EXCLUDED_LOGGERS = ["websockets", "httpx", "pydub"]
LOG_FILE = "audio_processor.log"


class ExcludeLoggerFilter(logging.Filter):
    def __init__(self, excluded_loggers):
        super().__init__()
        self.excluded_loggers = excluded_loggers

    def filter(self, record):
        return not any(record.name.startswith(logger) for logger in self.excluded_loggers)


def setup_logging():
    logger = logging.getLogger()
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Apply filter to exclude specific loggers
        filter = ExcludeLoggerFilter(EXCLUDED_LOGGERS)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        ch.addFilter(filter)
        logger.addHandler(ch)

        # File Handler
        fh = logging.FileHandler(LOG_FILE)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        fh.addFilter(filter)
        logger.addHandler(fh)

    # Optional: Prevent log messages from being propagated to the root logger
    logger.propagate = False

    return logger
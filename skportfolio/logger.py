import logging
from rich.logging import RichHandler


def get_logger(level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    handler = RichHandler()  # for beautiful coloured logging
    fmt = "%(message)s"
    formatter = logging.Formatter(fmt=fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

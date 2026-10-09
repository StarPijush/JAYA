"""
Logging Configuration

Centralized logging setup using Python standard library logging.
"""
import logging
import sys

from core.config.settings import get_settings


def setup_logging() -> None:
    settings = get_settings()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(log_level)
"""
Centralized structured logging configuration for MLOps Project 4.
Provides formatted stage-level logs, timestamping, and file + console handlers.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "mlops_pipeline", log_file: str = "pipeline.log") -> logging.Logger:
    """
    Configures and returns a structured logger with console and file output.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(log_format)
    logger.addHandler(c_handler)

    # File Handler
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f_handler = logging.FileHandler(log_path, encoding="utf-8")
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(log_format)
        logger.addHandler(f_handler)
    except Exception:
        pass

    return logger

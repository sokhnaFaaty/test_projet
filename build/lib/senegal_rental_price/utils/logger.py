"""Configuration centralisée du logging pour tout le projet."""

import logging
import sys
from pathlib import Path

_CONFIGURED = False
LOG_FILE = Path("logs/app.log")


def _configure_root_logger(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Retourne un logger configuré, à utiliser dans chaque module au lieu de print()."""
    _configure_root_logger(level)
    return logging.getLogger(name)

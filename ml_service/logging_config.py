from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import settings

_CONFIGURED = False


def setup_logging(logger_name: Optional[str] = None) -> logging.Logger:
    """
    Настраивает базовое логирование один раз и возвращает логгер.

    - stdout (Docker/K8s-friendly)
    - уровень из settings.log_level
    - единый формат
    """
    global _CONFIGURED

    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        logging.getLogger("uvicorn.error").setLevel(level)
        logging.getLogger("uvicorn.access").setLevel(level)

        _CONFIGURED = True

    return logging.getLogger(logger_name or "ml_service")
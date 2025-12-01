from __future__ import annotations

import logging
import sys

from .config import settings


def setup_logging() -> logging.Logger:
    """
    Настраивает базовый логгер для сервиса.

    - Логи пишутся в stdout (важно для Docker/Kubernetes).
    - Уровень логирования берётся из settings.log_level.
    - Дополнительно поднимаем уровень логов uvicorn, чтобы всё было консистентно.
    """
    logger = logging.getLogger("ml_service")

    # Чтобы не плодить хендлеры при повторных вызовах
    if logger.handlers:
        return logger

    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Настроим базовый уровень логирования для uvicorn
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)

    return logger

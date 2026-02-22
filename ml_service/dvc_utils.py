# ml_service/dvc_utils.py
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import settings
from .logging_config import setup_logging

logger = setup_logging(__name__)


def _run_dvc(*args: str) -> None:
    """
    Обёртка над вызовом dvc через subprocess.

    Бросает RuntimeError, если команда завершилась с ненулевым кодом.
    """
    cmd = ["dvc", *args]
    logger.info("Running DVC command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error(
            "DVC command failed (code=%s): %s\nstdout:\n%s\nstderr:\n%s",
            result.returncode,
            " ".join(cmd),
            result.stdout,
            result.stderr,
        )
        raise RuntimeError(
            f"DVC command {' '.join(args)} failed with exit code {result.returncode}"
        )


def dvc_add_and_push(path: Path) -> None:
    """
    Добавить файл датасета под управление DVC и передать в remote

    Ожидается, что:
      - dvc init уже выполнен,
      - remote настроен и выбран по умолчанию (dvc remote add ...; dvc remote default ...).

    Если DVC отключён (DVC_ENABLED=false), просто логируем и выходим.
    """
    if not settings.dvc_enabled:
        logger.info("DVC is disabled, skip add/push for %s", path)
        return

    rel_path = path.relative_to(settings.root_dir)
    logger.info("DVC add %s", rel_path)
    _run_dvc("add", str(rel_path))

    logger.info("DVC push %s.dvc", rel_path)
    _run_dvc("push", f"{rel_path}.dvc")
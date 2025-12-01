from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """
    Настройки сервиса.

    Все пути по умолчанию привязаны к корню репозитория:
    - SQLite-база: ml_service.db
    - Директория датасетов: data/datasets
    - Директория моделей: data/models
    """

    def __init__(self) -> None:
        # Корень проекта = папка, где лежит pyproject.toml
        self.root_dir: Path = Path(__file__).resolve().parent.parent

        # URL базы данных (по умолчанию SQLite-файл в корне проекта)
        default_db_path = self.root_dir / "ml_service.db"
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            f"sqlite:///{default_db_path}",
        )

        # Директория для датасетов
        default_datasets_dir = self.root_dir / "data" / "datasets"
        self.datasets_dir: Path = Path(
            os.getenv("DATASETS_DIR", str(default_datasets_dir))
        )

        # Директория для локального хранения моделей
        default_models_dir = self.root_dir / "data" / "models"
        self.models_dir: Path = Path(
            os.getenv("MODELS_DIR", str(default_models_dir))
        )

        # Уровень логов
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # Настройки ClearML
        self.clearml_enabled: bool = (
            os.getenv("CLEARML_ENABLED", "false").lower() == "true"
        )
        self.clearml_project: str = os.getenv(
            "CLEARML_PROJECT",
            "mlops_hw1",
        )
        self.clearml_task_prefix: str = os.getenv(
            "CLEARML_TASK_PREFIX",
            "train_",
        )
        # Куда ClearML будет складывать артефакты (обычно S3/Minio),
        # если в clearml.conf не задано. Можно оставить пустым.
        self.clearml_output_uri: str = os.getenv(
            "CLEARML_OUTPUT_URI",
            "",
        )

        # Создаём директории, если их ещё нет
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# Глобальный объект настроек, который будем импортировать из других модулей
settings = Settings()

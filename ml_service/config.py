from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """
    Настройки сервиса, читаемые из переменных окружения.
    """

    def __init__(self) -> None:
        self.root_dir: Path = Path(__file__).resolve().parent.parent

        default_db_path = self.root_dir / "ml_service.db"
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            f"sqlite:///{default_db_path}",
        )

        default_datasets_dir = self.root_dir / "data" / "datasets"
        self.datasets_dir: Path = Path(
            os.getenv("DATASETS_DIR", str(default_datasets_dir))
        )

        default_models_dir = self.root_dir / "data" / "models"
        self.models_dir: Path = Path(
            os.getenv("MODELS_DIR", str(default_models_dir))
        )

        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # ClearML
        self.clearml_enabled: bool = os.getenv("CLEARML_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self.clearml_project: str = os.getenv("CLEARML_PROJECT", "mlops-hw1")
        self.clearml_task_name_prefix: str = os.getenv(
            "CLEARML_TASK_NAME_PREFIX",
            "mlops-hw1",
        )
        self.clearml_output_uri: str = os.getenv(
            "CLEARML_OUTPUT_URI",
            "",
        )

        self.dvc_enabled: bool = os.getenv("DVC_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
        }

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
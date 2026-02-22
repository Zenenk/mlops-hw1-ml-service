from __future__ import annotations

from typing import Any, Dict, Optional

from .config import settings
from .logging_config import setup_logging

logger = setup_logging(__name__)


def start_clearml_task(
    *,
    project_name: str,
    task_name: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Стартует ClearML-задачу (эксперимент обучения).
    Возвращает Task или None, если ClearML выключен/недоступен.
    """
    if not settings.clearml_enabled:
        return None

    try:
        from clearml import Task
    except ImportError:
        logger.warning("ClearML не установлен, пропускаем интеграцию.")
        return None

    try:
        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,
        )

        if hyperparams:
            task.connect(hyperparams, name="hyperparameters")
        if metadata:
            task.connect(metadata, name="metadata")

        if settings.clearml_output_uri:
            task.set_output_uri(settings.clearml_output_uri)

        logger.info("ClearML task started: name=%s id=%s", task.name, task.id)
        return task
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to start ClearML task: %s", exc)
        return None


def log_model_to_clearml(
    *,
    task: Any,
    model_path: str,
    model_name: str,
) -> Optional[str]:
    """
    Загружает веса модели в ClearML и возвращает id модели.
    """
    if task is None:
        return None

    try:
        from clearml import OutputModel
    except ImportError:
        logger.warning("ClearML не установлен, не можем загрузить модель.")
        return None

    try:
        output_model = OutputModel(task=task, name=model_name)
        output_model.update_weights(weights_filename=model_path)
        logger.info("Model uploaded to ClearML: name=%s id=%s", model_name, output_model.id)
        return output_model.id
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to upload model to ClearML: %s", exc)
        return None


def close_clearml_task(task: Any, *, failed: bool = False) -> None:
    """
    Закрывает ClearML-задачу.
    Параметр failed нужен, потому что services.py его передаёт.
    """
    if task is None:
        return

    try:
        if failed:
            try:
                task.mark_failed()
            except Exception:  # noqa: BLE001
                pass
        task.close()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to close ClearML task: %s", exc)
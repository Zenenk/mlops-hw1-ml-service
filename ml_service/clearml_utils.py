from __future__ import annotations

from typing import Any, Dict, Optional

from .config import settings
from .logging_config import setup_logging

logger = setup_logging()


def start_clearml_task(
    model_name: str,
    model_class: str,
    dataset_id: int,
    hyperparams: Dict[str, Any],
) -> Optional[Any]:
    """
    Стартует ClearML-задачу для обучения модели.

    Возвращает объект Task или None, если ClearML выключен или недоступен.
    """
    if not settings.clearml_enabled:
        return None

    try:
        from clearml import Task
    except ImportError:
        logger.warning("ClearML не установлен, пропускаем интеграцию с ClearML.")
        return None

    try:
        task_name = f"{settings.clearml_task_prefix}{model_name}"
        task = Task.init(
            project_name=settings.clearml_project,
            task_name=task_name,
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,
        )

        # Логируем гиперпараметры и метаданные
        task.connect(hyperparams, name="hyperparameters")
        task.connect(
            {
                "dataset_id": dataset_id,
                "model_class": model_class,
            },
            name="metadata",
        )

        if settings.clearml_output_uri:
            task.set_output_uri(settings.clearml_output_uri)

        logger.info(
            "ClearML task started: name=%s id=%s",
            task.name,
            task.id,
        )
        return task
    except Exception as exc:
        logger.exception("Failed to start ClearML task: %s", exc)
        return None


def log_model_to_clearml(
    task: Any,
    model_path: str,
    model_name: str,
) -> Optional[str]:
    """
    Загружает веса модели в ClearML и возвращает идентификатор модели.

    Если task == None или ClearML недоступен, возвращает None.
    """
    if task is None:
        return None

    try:
        from clearml import OutputModel
    except ImportError:
        logger.warning("ClearML не установлен, не можем загрузить модель в ClearML.")
        return None

    try:
        output_model = OutputModel(
            task=task,
            name=model_name,
        )
        output_model.update_weights(weights_filename=model_path)
        logger.info(
            "Model '%s' uploaded to ClearML with id=%s",
            model_name,
            output_model.id,
        )
        return output_model.id
    except Exception as exc:
        logger.exception("Failed to upload model to ClearML: %s", exc)
        return None


def close_clearml_task(task: Any) -> None:
    """
    Аккуратно закрывает ClearML-задачу, если она есть.
    """
    if task is None:
        return
    try:
        task.close()
    except Exception as exc:
        logger.exception("Failed to close ClearML task: %s", exc)

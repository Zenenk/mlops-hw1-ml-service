from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .clearml_utils import close_clearml_task, log_model_to_clearml, start_clearml_task
from .config import settings
from .db_models import Dataset, Model
from .dvc_utils import dvc_add_and_push
from .logging_config import setup_logging
from .model_registry import UnsupportedModelError, create_model_instance, get_available_model_classes

logger = setup_logging(__name__)


def _save_dataset_file(original_name: str, content: bytes) -> Path:
    datasets_dir = settings.datasets_dir
    datasets_dir.mkdir(parents=True, exist_ok=True)

    unique_prefix = uuid.uuid4().hex
    stored_name = f"{unique_prefix}_{original_name}"
    stored_path = datasets_dir / stored_name

    stored_path.write_bytes(content)
    logger.info("Dataset saved to %s", stored_path)
    return stored_path


def _load_dataset_xy(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    ext = path.suffix.lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Неподдерживаемый формат датасета: {ext}")
    except FileNotFoundError as exc:
        raise ValueError(f"Файл датасета не найден: {path}") from exc

    if "target" not in df.columns:
        raise ValueError("В датасете нет столбца 'target'")

    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()
    logger.info("Loaded dataset %s: X shape=%s, y shape=%s", path, X.shape, y.shape)
    return X, y


def _save_model_to_disk(model_obj: Any, model_class: str) -> Path:
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{model_class}.joblib"
    model_path = models_dir / unique_name
    joblib.dump(model_obj, model_path)
    logger.info("Model saved to %s", model_path)
    return model_path


def _resolve_hyperparams(model_class: str, user_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    registry = get_available_model_classes()
    if model_class not in registry:
        raise UnsupportedModelError(
            f"Неизвестный класс модели: {model_class!r}. Доступные: {', '.join(registry.keys())}"
        )
    default_params = dict(registry[model_class].get("default_hyperparams") or {})
    merged = {**default_params, **(user_hyperparams or {})}
    return merged


def upload_dataset_service(
    db: Session,
    *,
    filename: str,
    content: bytes,
    description: str | None = None,
) -> Dataset:
    stored_path = _save_dataset_file(filename, content)

    try:
        dvc_add_and_push(stored_path)
    except Exception:  # noqa: BLE001
        logger.exception("DVC add/push failed for %s", stored_path)

    dataset = Dataset(
        name=stored_path.name,
        path=str(stored_path),
        description=description,
        version=None,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def list_datasets_service(db: Session) -> List[Dataset]:
    return db.query(Dataset).order_by(Dataset.id.asc()).all()


def delete_dataset_service(db: Session, dataset_id: int) -> None:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise ValueError(f"Dataset id={dataset_id} not found")

    try:
        if dataset.path and os.path.exists(dataset.path):
            os.remove(dataset.path)
            logger.info("Removed dataset file %s", dataset.path)
    except OSError:
        logger.exception("Failed to remove dataset file %s", dataset.path)

    db.delete(dataset)
    db.commit()


def list_models_service(db: Session) -> List[Model]:
    return db.query(Model).order_by(Model.id.asc()).all()


def _train_model_core(
    *,
    model_name: str,
    model_class: str,
    dataset_id: int,
    X: np.ndarray,
    y: np.ndarray,
    hyperparams: Dict[str, Any],
) -> Tuple[Any, Path, str | None]:
    try:
        model_obj = create_model_instance(model_class, hyperparams)
    except UnsupportedModelError:
        raise
    except TypeError as exc:
        raise ValueError(f"Некорректные гиперпараметры: {exc}") from exc

    clearml_task = None
    clearml_model_id: str | None = None

    if settings.clearml_enabled:
        clearml_task = start_clearml_task(
            project_name=settings.clearml_project,
            task_name=f"{settings.clearml_task_name_prefix} - train {model_name}",
            hyperparams=hyperparams,
            metadata={
                "model_name": model_name,
                "model_class": model_class,
                "dataset_id": dataset_id,
            },
        )

    try:
        model_obj.fit(X, y)
    except Exception as exc:
        if clearml_task is not None:
            close_clearml_task(clearml_task, failed=True)
        raise RuntimeError(f"Ошибка обучения модели: {exc}") from exc

    model_path = _save_model_to_disk(model_obj, model_class)

    if clearml_task is not None:
        clearml_model_id = log_model_to_clearml(
            task=clearml_task,
            model_path=str(model_path),
            model_name=model_name,
        )
        close_clearml_task(clearml_task, failed=False)

    return model_obj, model_path, clearml_model_id


def train_model_service(
    db: Session,
    *,
    name: str,
    model_class: str,
    dataset_id: int,
    hyperparams: Dict[str, Any] | None = None,
) -> Model:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise ValueError(f"Dataset id={dataset_id} not found")

    merged_hyperparams = _resolve_hyperparams(model_class, hyperparams or {})

    X, y = _load_dataset_xy(dataset.path)
    _, model_path, clearml_model_id = _train_model_core(
        model_name=name,
        model_class=model_class,
        dataset_id=dataset.id,
        X=X,
        y=y,
        hyperparams=merged_hyperparams,
    )

    model_row = Model(
        name=name,
        model_class=model_class,
        dataset_id=dataset.id,
        status="trained",
        hyperparams=merged_hyperparams,
        clearml_model_id=clearml_model_id,
        local_path=str(model_path),
    )
    db.add(model_row)
    db.commit()
    db.refresh(model_row)
    return model_row


def retrain_model_service(
    db: Session,
    *,
    model_id: int,
    model_class: str | None = None,
    hyperparams: Dict[str, Any] | None = None,
) -> Model:
    model_row = db.query(Model).filter(Model.id == model_id).first()
    if model_row is None:
        raise ValueError(f"Model id={model_id} not found")

    dataset = db.query(Dataset).filter(Dataset.id == model_row.dataset_id).first()
    if dataset is None:
        raise RuntimeError(f"Dataset id={model_row.dataset_id} not found for model id={model_id}")

    model_class_eff = model_class or model_row.model_class

    # Если пользователь не передал параметры (или передал пустые), переиспользуем прошлые.
    if hyperparams and len(hyperparams) > 0:
        merged_hyperparams = _resolve_hyperparams(model_class_eff, hyperparams)
    else:
        merged_hyperparams = _resolve_hyperparams(model_class_eff, model_row.hyperparams or {})

    X, y = _load_dataset_xy(dataset.path)
    _, model_path, clearml_model_id = _train_model_core(
        model_name=model_row.name,
        model_class=model_class_eff,
        dataset_id=dataset.id,
        X=X,
        y=y,
        hyperparams=merged_hyperparams,
    )

    model_row.model_class = model_class_eff
    model_row.local_path = str(model_path)
    model_row.status = "trained"
    model_row.clearml_model_id = clearml_model_id
    model_row.hyperparams = merged_hyperparams

    db.commit()
    db.refresh(model_row)
    return model_row


def delete_model_service(db: Session, model_id: int) -> None:
    model_row = db.query(Model).filter(Model.id == model_id).first()
    if model_row is None:
        raise ValueError(f"Model id={model_id} not found")

    model_row.status = "deleted"
    db.commit()

    try:
        if model_row.local_path and os.path.exists(model_row.local_path):
            os.remove(model_row.local_path)
            logger.info("Removed model file %s", model_row.local_path)
    except OSError:
        logger.exception("Failed to remove model file %s", model_row.local_path)


def predict_service(
    db: Session,
    *,
    model_id: int,
    features: List[List[float]],
) -> List[int]:
    model_row = db.query(Model).filter(Model.id == model_id).first()
    if model_row is None:
        raise ValueError(f"Model id={model_id} not found")

    if model_row.status != "trained":
        raise ValueError(f"Модель id={model_id} находится в статусе '{model_row.status}'")

    if not model_row.local_path or not os.path.exists(model_row.local_path):
        raise RuntimeError(f"Файл модели для id={model_id} не найден: {model_row.local_path}")

    model_obj = joblib.load(model_row.local_path)
    X = np.asarray(features, dtype=float)
    preds = model_obj.predict(X)
    return [int(p) for p in preds]
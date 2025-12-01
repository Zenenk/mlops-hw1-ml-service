from __future__ import annotations

import os
import uuid
from typing import Iterator, List

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .clearml_utils import (close_clearml_task, log_model_to_clearml,
                            start_clearml_task)
from .config import settings
from .db_models import Base, Dataset, Model
from .logging_config import setup_logging
from .model_registry import (UnsupportedModelError, create_model_instance,
                             get_available_model_classes)
from .schemas import (DatasetInfo, DatasetListResponse, HealthResponse,
                      ModelClassesResponse, ModelClassInfo, ModelInfo,
                      ModelListResponse, PredictRequest, PredictResponse,
                      RetrainModelRequest, TrainModelRequest,
                      TrainModelResponse)

# Инициализация БД и логгера
logger = setup_logging()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Создаём таблицы при старте модуля (однократно)
Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """
    Зависимость FastAPI для выдачи сессии БД в эндпоинты.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Вспомогательные функции
def load_dataset_to_numpy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает датасет из CSV/JSON и возвращает (X, y).

    Ожидается, что в датасете есть столбец 'target', который
    используется как целевая переменная.
    """
    logger.info("Loading dataset from %s", path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("Поддерживаются только форматы CSV и JSON")

    if "target" not in df.columns:
        raise ValueError("В датасете должен быть столбец 'target'")

    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()
    return X, y



# Инициализация FastAPI-приложения
app = FastAPI(
    title="ML Homework Service",
    version="0.1.0",
    description=(
        "Сервис для обучения и инференса ML-моделей "
        "(логистическая регрессия и случайный лес) "
        "с поддержкой нескольких моделей и датасетов."
    ),
)

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при желании можно ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Эндпоинты статуса и классов моделей
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Эндпоинт для проверки статуса сервиса.
    """
    logger.info("Health check called")
    return HealthResponse(status="ok", version="0.1.0")

@app.get("/model_classes", response_model=List[ModelClassInfo], tags=["models"])
def list_model_classes_flat() -> List[ModelClassInfo]:
    """
    Вернуть список доступных классов моделей для обучения.
    """
    logger.info("Listing available model classes")
    classes_dict = get_available_model_classes()
    return [
        ModelClassInfo(
            id=key,
            description=meta["description"],
            default_hyperparams=meta["default_hyperparams"],
        )
        for key, meta in classes_dict.items()
    ]



@app.get("/model-classes", response_model=ModelClassesResponse)
def list_model_classes() -> ModelClassesResponse:
    """
    Возвращает список доступных для обучения классов моделей и их
    дефолтные гиперпараметры.
    """
    logger.info("Listing model classes")
    classes_dict = get_available_model_classes()
    classes: list[ModelClassInfo] = []
    for key, meta in classes_dict.items():
        classes.append(
            ModelClassInfo(
                id=key,
                description=meta["description"],
                default_hyperparams=meta["default_hyperparams"],
            )
        )
    return ModelClassesResponse(classes=classes)



# Эндпоинты для работы с датасетами
@app.post(
    "/datasets",
    response_model=DatasetInfo,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: UploadFile = File(...),
    description: str = "",
    db: Session = Depends(get_db),
) -> DatasetInfo:
    """
    Загружает новый датасет (CSV или JSON) и создаёт запись в БД.

    Требование: датасет должен содержать столбец 'target'.
    """
    logger.info("Uploading dataset: %s", file.filename)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".csv", ".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поддерживаются только файлы с расширением .csv или .json",
        )

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest_path = settings.datasets_dir / unique_name

    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
    except OSError as exc:
        logger.exception("Failed to save dataset file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось сохранить файл датасета: {exc}",
        )

    dataset = Dataset(
        name=unique_name,
        path=str(dest_path),
        description=description or None,
        version=None,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    logger.info("Dataset stored: id=%s path=%s", dataset.id, dataset.path)

    return DatasetInfo(
        id=dataset.id,
        name=dataset.name,
        path=dataset.path,
        description=dataset.description,
        version=dataset.version,
    )


@app.get("/datasets", response_model=DatasetListResponse)
def list_datasets(db: Session = Depends(get_db)) -> DatasetListResponse:
    """
    Возвращает список всех загруженных датасетов.
    """
    logger.info("Listing datasets")
    items = db.query(Dataset).order_by(Dataset.id.asc()).all()

    datasets = [
        DatasetInfo(
            id=d.id,
            name=d.name,
            path=d.path,
            description=d.description,
            version=d.version,
        )
        for d in items
    ]
    return DatasetListResponse(datasets=datasets)


@app.delete("/datasets/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    """
    Удаляет датасет и физически файл на диске.

    В БД настроен каскад для связанных моделей, поэтому вместе
    с датасетом будут удалены и модели, обученные на нём.
    """
    logger.info("Deleting dataset id=%s", dataset_id)
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    # Сначала пытаемся удалить файл
    try:
        if os.path.exists(dataset.path):
            os.remove(dataset.path)
    except OSError:
        logger.exception("Failed to remove dataset file")

    # Затем удаляем запись из БД (модели удалятся каскадно)
    db.delete(dataset)
    db.commit()

    return {"status": "deleted"}



# Эндпоинты для работы с моделями
@app.post(
    "/models/train",
    response_model=TrainModelResponse,
    status_code=status.HTTP_201_CREATED,
)
def train_model(
    body: TrainModelRequest,
    db: Session = Depends(get_db),
) -> TrainModelResponse:
    """
    Обучает новую модель на указанном датасете и сохраняет её на диск
    и в БД. Гиперпараметры передаются в теле запроса.

    При включённом ClearML каждое обучение регистрируется как отдельный
    эксперимент, а веса модели загружаются как отдельная модель в ClearML.
    """
    logger.info(
        "Train model requested: name=%s class=%s dataset_id=%s",
        body.name,
        body.model_class,
        body.dataset_id,
    )

    dataset = db.query(Dataset).filter(Dataset.id == body.dataset_id).first()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )

    try:
        X, y = load_dataset_to_numpy(dataset.path)
    except Exception as exc:
        logger.exception("Failed to load dataset for training")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # Проверяем, что класс модели поддерживается
    classes_dict = get_available_model_classes()
    if body.model_class not in classes_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неизвестный класс модели: {body.model_class}",
        )

    try:
        model_obj = create_model_instance(body.model_class, body.hyperparams)
    except UnsupportedModelError as exc:
        logger.exception("Unsupported model class")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except TypeError as exc:
        logger.exception("Invalid hyperparameters for model class %s", body.model_class)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Некорректные гиперпараметры: {exc}",
        )

    # Стартуем ClearML-задачу (если включена)
    clearml_task = start_clearml_task(
        model_name=body.name,
        model_class=body.model_class,
        dataset_id=dataset.id,
        hyperparams=body.hyperparams,
    )

    # Обучаем модель
    try:
        logger.info("Fitting model '%s' on dataset %s", body.name, dataset.id)
        model_obj.fit(X, y)
    except Exception as exc:
        logger.exception("Model training failed")
        # Можно пометить задачу как failed, но здесь просто закрываем
        close_clearml_task(clearml_task)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обучения модели: {exc}",
        )

    # Сохраняем веса на диск
    unique_filename = f"{uuid.uuid4().hex}_{body.model_class}.joblib"
    model_path = settings.models_dir / unique_filename
    try:
        joblib.dump(model_obj, model_path)
    except Exception as exc:
        logger.exception("Failed to save model file")
        close_clearml_task(clearml_task)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось сохранить модель: {exc}",
        )

    # Логируем модель в ClearML (если задача существует)
    clearml_model_id = log_model_to_clearml(
        task=clearml_task,
        model_path=str(model_path),
        model_name=body.name,
    )
    close_clearml_task(clearml_task)

    model_row = Model(
        name=body.name,
        model_class=body.model_class,
        dataset_id=dataset.id,
        status="trained",
        hyperparams=body.hyperparams,
        clearml_model_id=clearml_model_id,
        local_path=str(model_path),
    )
    db.add(model_row)
    db.commit()
    db.refresh(model_row)

    logger.info("Model registered in DB with id=%s", model_row.id)

    return TrainModelResponse(model_id=model_row.id, status=model_row.status)



@app.get("/models", response_model=ModelListResponse)
def list_models(
    db: Session = Depends(get_db),
) -> ModelListResponse:
    """
    Возвращает список всех моделей (включая, при желании, помеченные deleted).
    """
    logger.info("Listing models")
    items = (
        db.query(Model)
        .order_by(Model.id.asc())
        .all()
    )

    models = [
        ModelInfo(
            id=m.id,
            name=m.name,
            model_class=m.model_class,
            dataset_id=m.dataset_id,
            status=m.status,
            hyperparams=m.hyperparams,
            clearml_model_id=m.clearml_model_id,
        )
        for m in items
    ]
    return ModelListResponse(models=models)


@app.post(
    "/models/{model_id}/predict",
    response_model=PredictResponse,
)
def predict(
    model_id: int,
    body: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """
    Возвращает предсказания для указанной модели и батча объектов.
    """
    logger.info("Predict request for model id=%s", model_id)
    model_row = db.query(Model).filter(Model.id == model_id).first()
    if model_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    if model_row.status != "trained":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Модель находится в статусе '{model_row.status}', инференс невозможен",
        )

    if not os.path.exists(model_row.local_path):
        logger.error("Model file not found: %s", model_row.local_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Файл модели не найден на диске",
        )

    try:
        model_obj = joblib.load(model_row.local_path)
    except Exception as exc:
        logger.exception("Failed to load model file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось загрузить модель: {exc}",
        )

    X = np.array(body.features, dtype=float)
    try:
        preds = model_obj.predict(X)
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка инференса: {exc}",
        )

    predictions = [int(v) for v in preds]
    logger.info("Predicted %s samples", len(predictions))

    return PredictResponse(predictions=predictions)


@app.delete("/models/{model_id}")
def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
):
    """
    Помечает модель как удалённую и пытается удалить файл с диска.
    """
    logger.info("Deleting model id=%s", model_id)
    model_row = db.query(Model).filter(Model.id == model_id).first()
    if model_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Помечаем как удалённую
    model_row.status = "deleted"
    db.commit()

    # Пытаемся удалить файл
    try:
        if os.path.exists(model_row.local_path):
            os.remove(model_row.local_path)
    except OSError:
        logger.exception("Failed to remove model file")

    return {"status": "deleted"}


@app.post(
    "/models/{model_id}/retrain",
    response_model=TrainModelResponse,
)
def retrain_model(
    model_id: int,
    body: RetrainModelRequest,
    db: Session = Depends(get_db),
) -> TrainModelResponse:
    """
    Переобучает уже существующую модель на том же датасете,
    но с новыми гиперпараметрами и (при необходимости) другим
    классом модели. При включённом ClearML создаётся новый
    эксперимент и новая модель в ClearML.
    """
    logger.info(
        "Retrain model id=%s with class=%s",
        model_id,
        body.model_class,
    )

    model_row = db.query(Model).filter(Model.id == model_id).first()
    if model_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    dataset = db.query(Dataset).filter(Dataset.id == model_row.dataset_id).first()
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found for this model",
        )

    try:
        X, y = load_dataset_to_numpy(dataset.path)
    except Exception as exc:
        logger.exception("Failed to load dataset for retraining")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    classes_dict = get_available_model_classes()
    if body.model_class not in classes_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неизвестный класс модели: {body.model_class}",
        )

    try:
        model_obj = create_model_instance(body.model_class, body.hyperparams)
    except UnsupportedModelError as exc:
        logger.exception("Unsupported model class in retrain")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except TypeError as exc:
        logger.exception(
            "Invalid hyperparameters for model class %s in retrain",
            body.model_class,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Некорректные гиперпараметры: {exc}",
        )

    # Стартуем ClearML-задачу для переобучения
    clearml_task = start_clearml_task(
        model_name=model_row.name,
        model_class=body.model_class,
        dataset_id=dataset.id,
        hyperparams=body.hyperparams,
    )

    # Переобучаем модель
    try:
        logger.info("Fitting model id=%s on dataset %s", model_row.id, dataset.id)
        model_obj.fit(X, y)
    except Exception as exc:
        logger.exception("Model retraining failed")
        close_clearml_task(clearml_task)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка переобучения модели: {exc}",
        )

    # Сохраняем новые веса
    unique_filename = f"{uuid.uuid4().hex}_{body.model_class}.joblib"
    model_path = settings.models_dir / unique_filename
    try:
        joblib.dump(model_obj, model_path)
    except Exception as exc:
        logger.exception("Failed to save retrained model file")
        close_clearml_task(clearml_task)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось сохранить модель: {exc}",
        )

    # Логируем новую версию модели в ClearML
    clearml_model_id = log_model_to_clearml(
        task=clearml_task,
        model_path=str(model_path),
        model_name=model_row.name,
    )
    close_clearml_task(clearml_task)

    model_row.model_class = body.model_class
    model_row.hyperparams = body.hyperparams
    model_row.local_path = str(model_path)
    model_row.status = "trained"
    model_row.clearml_model_id = clearml_model_id
    db.commit()
    db.refresh(model_row)

    logger.info("Model id=%s retrained successfully", model_row.id)

    return TrainModelResponse(
        model_id=model_row.id,
        status=model_row.status,
    )


# Возможность запустить приложение напрямую: python -m ml_service.api_rest
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ml_service.api_rest:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

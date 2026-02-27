from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .db_models import Base
from .logging_config import setup_logging
from .model_registry import UnsupportedModelError, get_available_model_classes
from .schemas import (
    DatasetInfo,
    DatasetListResponse,
    HealthResponse,
    ModelClassInfo,
    ModelClassesResponse,
    ModelInfo,
    ModelListResponse,
    PredictBatchResponse,
)
from .services import (
    delete_dataset_service,
    delete_model_service,
    list_datasets_service,
    list_models_service,
    batch_predict_service,
    upload_dataset_service,
)

logger = setup_logging(__name__)

app = FastAPI(
    title="ML Homework 1 Service",
    description="REST API для обучения и инференса простых ML-моделей.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine_kwargs = {"future": True}
if settings.database_url.startswith("sqlite"):
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        **engine_kwargs,
    )
else:
    engine = create_engine(settings.database_url, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health", response_model=HealthResponse, tags=["service"])
def healthcheck() -> HealthResponse:
    logger.info("Health check called")
    return HealthResponse(status="ok", version="0.1.0")


def _model_classes_payload() -> ModelClassesResponse:
    registry = get_available_model_classes()
    items = [
        ModelClassInfo(
            id=model_id,
            description=meta["description"],
            default_hyperparams=meta.get("default_hyperparams") or {},
        )
        for model_id, meta in registry.items()
    ]
    return ModelClassesResponse(classes=items)


@app.get("/model-classes", response_model=ModelClassesResponse, tags=["models"])
def list_model_classes() -> ModelClassesResponse:
    logger.info("Listing model classes")
    return _model_classes_payload()


@app.get("/model_classes", response_model=ModelClassesResponse, tags=["models"])
def list_model_classes_alias() -> ModelClassesResponse:
    logger.info("Listing model classes (alias)")
    return _model_classes_payload()


@app.get("/datasets", response_model=DatasetListResponse, tags=["datasets"])
def list_datasets(db: Session = Depends(get_db)) -> DatasetListResponse:
    logger.info("Listing datasets")
    datasets = list_datasets_service(db)
    items = [
        DatasetInfo(
            id=d.id,
            name=d.name,
            path=d.path,
            description=d.description,
            version=d.version,
        )
        for d in datasets
    ]
    return DatasetListResponse(datasets=items)


@app.post("/datasets", response_model=DatasetInfo, tags=["datasets"])
async def upload_dataset(
    file: UploadFile = File(...),
    description: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> DatasetInfo:
    logger.info("Uploading dataset: filename=%s", file.filename)
    content = await file.read()

    try:
        dataset = upload_dataset_service(
            db,
            filename=file.filename,
            content=content,
            description=description,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to upload dataset")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DatasetInfo(
        id=dataset.id,
        name=dataset.name,
        path=dataset.path,
        description=dataset.description,
        version=dataset.version,
    )


@app.delete("/datasets/{dataset_id}", status_code=204, tags=["datasets"])
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)) -> None:
    logger.info("Deleting dataset id=%s", dataset_id)
    try:
        delete_dataset_service(db, dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/models", response_model=ModelListResponse, tags=["models"])
def list_models(db: Session = Depends(get_db)) -> ModelListResponse:
    logger.info("Listing models")
    models = list_models_service(db)
    items = [
        ModelInfo(
            id=m.id,
            name=m.name,
            model_class=m.model_class,
            dataset_id=m.dataset_id,
            status=m.status,
            hyperparams=m.hyperparams or {},
            clearml_model_id=m.clearml_model_id,
        )
        for m in models
    ]
    return ModelListResponse(models=items)


@app.post("/models/train", response_model=TrainModelResponse, status_code=201, tags=["models"])
def train_model(
    req: TrainModelRequest,
    db: Session = Depends(get_db),
) -> TrainModelResponse:
    logger.info(
        "Train model: name=%s, model_class=%s, dataset_id=%s",
        req.name,
        req.model_class,
        req.dataset_id,
    )
    try:
        model_row = train_model_service(
            db,
            name=req.name,
            model_class=req.model_class,
            dataset_id=req.dataset_id,
            hyperparams=req.hyperparams,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TrainModelResponse(model_id=model_row.id, status="trained")


@app.delete("/models/{model_id}", status_code=204, tags=["models"])
def delete_model(model_id: int, db: Session = Depends(get_db)) -> None:
    logger.info("Delete model id=%s", model_id)
    try:
        delete_model_service(db, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/models/{model_id}/predict-batch", response_model=PredictBatchResponse)
async def predict_batch(
    model_id: int,
    file: UploadFile = File(...),
    has_header: bool = Query(True),
    db: Session = Depends(get_db),
) -> PredictBatchResponse:
    try:
        return batch_predict_service(db, model_id, file, has_header)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

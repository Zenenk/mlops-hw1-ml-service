from __future__ import annotations

import json
import os
from concurrent import futures
from typing import Iterator

import grpc
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from . import ml_service_pb2, ml_service_pb2_grpc
from .clearml_utils import (close_clearml_task, log_model_to_clearml,
                            start_clearml_task)
from .config import settings
from .db_models import Base, Dataset, Model
from .logging_config import setup_logging
from .model_registry import (UnsupportedModelError, create_model_instance,
                             get_available_model_classes)

logger = setup_logging()

# Настройка БД
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# Вспомогательная функция для загрузки датасета
def load_dataset_to_numpy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает датасет из CSV/JSON и возвращает (X, y).

    Требование: в датасете должен быть столбец 'target'.
    """
    logger.info("Loading dataset for gRPC from %s", path)
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



# Реализация gRPC-сервиса
class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    """
    Реализация методов gRPC-сервиса, опирающаяся на ту же БД,
    что и REST API.
    """

    #HealthCheck
    def HealthCheck(self, request, context):
        logger.info("gRPC HealthCheck called")
        return ml_service_pb2.HealthResponse(
            status="ok",
            version="0.1.0",
        )

    #ListModelClasses
    def ListModelClasses(self, request, context):
        logger.info("gRPC ListModelClasses called")
        classes_dict = get_available_model_classes()
        resp = ml_service_pb2.ModelClassesResponse()
        for key, meta in classes_dict.items():
            info = ml_service_pb2.ModelClassInfo(
                id=key,
                description=meta["description"],
                default_hyperparams_json=json.dumps(meta["default_hyperparams"]),
            )
            resp.classes.append(info)
        return resp

    #ListDatasets
    def ListDatasets(self, request, context):
        logger.info("gRPC ListDatasets called")
        db = SessionLocal()
        try:
            items = db.query(Dataset).order_by(Dataset.id.asc()).all()
            resp = ml_service_pb2.DatasetListResponse()
            for d in items:
                info = ml_service_pb2.DatasetInfo(
                    id=d.id,
                    name=d.name,
                    path=d.path,
                    description=d.description or "",
                    version=d.version or "",
                )
                resp.datasets.append(info)
            return resp
        finally:
            db.close()

    #ListModels
    def ListModels(self, request, context):
        logger.info("gRPC ListModels called")
        db = SessionLocal()
        try:
            items = db.query(Model).order_by(Model.id.asc()).all()
            resp = ml_service_pb2.ModelListResponse()
            for m in items:
                info = ml_service_pb2.ModelInfo(
                    id=m.id,
                    name=m.name,
                    model_class=m.model_class,
                    dataset_id=m.dataset_id,
                    status=m.status,
                    hyperparams_json=json.dumps(m.hyperparams),
                    clearml_model_id=m.clearml_model_id or "",
                )
                resp.models.append(info)
            return resp
        finally:
            db.close()

    #TrainModel
    def TrainModel(self, request, context):
        logger.info(
            "gRPC TrainModel: name=%s class=%s dataset_id=%s",
            request.name,
            request.model_class,
            request.dataset_id,
        )

        db = SessionLocal()
        try:
            dataset = (
                db.query(Dataset)
                .filter(Dataset.id == request.dataset_id)
                .first()
            )
            if dataset is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Dataset not found")
                return ml_service_pb2.TrainModelResponse()

            try:
                X, y = load_dataset_to_numpy(dataset.path)
            except Exception as exc:
                logger.exception("Failed to load dataset for training (gRPC)")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()

            classes_dict = get_available_model_classes()
            if request.model_class not in classes_dict:
                msg = f"Неизвестный класс модели: {request.model_class}"
                logger.error(msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return ml_service_pb2.TrainModelResponse()

            try:
                hyperparams = json.loads(request.hyperparams_json)
            except json.JSONDecodeError as exc:
                msg = f"Некорректный JSON гиперпараметров: {exc}"
                logger.error(msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return ml_service_pb2.TrainModelResponse()

            try:
                model_obj = create_model_instance(request.model_class, hyperparams)
            except UnsupportedModelError as exc:
                logger.exception("Unsupported model class in gRPC TrainModel")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except TypeError as exc:
                logger.exception("Invalid hyperparameters for %s", request.model_class)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Некорректные гиперпараметры: {exc}")
                return ml_service_pb2.TrainModelResponse()

            # ClearML-задача
            clearml_task = start_clearml_task(
                model_name=request.name,
                model_class=request.model_class,
                dataset_id=dataset.id,
                hyperparams=hyperparams,
            )

            # Обучаем модель
            try:
                logger.info(
                    "gRPC fitting model '%s' on dataset %s",
                    request.name,
                    dataset.id,
                )
                model_obj.fit(X, y)
            except Exception as exc:
                logger.exception("Model training failed (gRPC)")
                close_clearml_task(clearml_task)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Ошибка обучения модели: {exc}")
                return ml_service_pb2.TrainModelResponse()

            # Сохраняем веса
            unique_filename = f"grpc_{request.model_class}_{dataset.id}.joblib"
            model_path = settings.models_dir / unique_filename
            try:
                joblib.dump(model_obj, model_path)
            except Exception as exc:
                logger.exception("Failed to save model file (gRPC)")
                close_clearml_task(clearml_task)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Не удалось сохранить модель: {exc}")
                return ml_service_pb2.TrainModelResponse()

            # Логируем модель в ClearML
            clearml_model_id = log_model_to_clearml(
                task=clearml_task,
                model_path=str(model_path),
                model_name=request.name,
            )
            close_clearml_task(clearml_task)

            model_row = Model(
                name=request.name,
                model_class=request.model_class,
                dataset_id=dataset.id,
                status="trained",
                hyperparams=hyperparams,
                clearml_model_id=clearml_model_id,
                local_path=str(model_path),
            )
            db.add(model_row)
            db.commit()
            db.refresh(model_row)

            logger.info("gRPC model registered id=%s", model_row.id)

            return ml_service_pb2.TrainModelResponse(
                model_id=model_row.id,
                status=model_row.status,
            )
        finally:
            db.close()


    # RetrainModel
    def RetrainModel(self, request, context):
        logger.info("gRPC RetrainModel: model_id=%s", request.model_id)
        db = SessionLocal()
        try:
            model_row = (
                db.query(Model)
                .filter(Model.id == request.model_id)
                .first()
            )
            if model_row is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Model not found")
                return ml_service_pb2.TrainModelResponse()

            dataset = (
                db.query(Dataset)
                .filter(Dataset.id == model_row.dataset_id)
                .first()
            )
            if dataset is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Dataset not found for this model")
                return ml_service_pb2.TrainModelResponse()

            try:
                X, y = load_dataset_to_numpy(dataset.path)
            except Exception as exc:
                logger.exception("Failed to load dataset for retraining (gRPC)")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()

            classes_dict = get_available_model_classes()
            if request.model_class not in classes_dict:
                msg = f"Неизвестный класс модели: {request.model_class}"
                logger.error(msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return ml_service_pb2.TrainModelResponse()

            try:
                hyperparams = json.loads(request.hyperparams_json)
            except json.JSONDecodeError as exc:
                msg = f"Некорректный JSON гиперпараметров: {exc}"
                logger.error(msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return ml_service_pb2.TrainModelResponse()

            try:
                model_obj = create_model_instance(request.model_class, hyperparams)
            except UnsupportedModelError as exc:
                logger.exception("Unsupported model class in gRPC RetrainModel")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except TypeError as exc:
                logger.exception(
                    "Invalid hyperparameters for %s in retrain",
                    request.model_class,
                )
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Некорректные гиперпараметры: {exc}")
                return ml_service_pb2.TrainModelResponse()

            # ClearML-задача
            clearml_task = start_clearml_task(
                model_name=model_row.name,
                model_class=request.model_class,
                dataset_id=dataset.id,
                hyperparams=hyperparams,
            )

            # Переобучаем
            try:
                logger.info(
                    "gRPC fitting model id=%s on dataset %s",
                    model_row.id,
                    dataset.id,
                )
                model_obj.fit(X, y)
            except Exception as exc:
                logger.exception("Model retraining failed (gRPC)")
                close_clearml_task(clearml_task)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Ошибка переобучения модели: {exc}")
                return ml_service_pb2.TrainModelResponse()

            # Сохраняем новые веса
            unique_filename = f"grpc_retrain_{request.model_class}_{dataset.id}.joblib"
            model_path = settings.models_dir / unique_filename
            try:
                joblib.dump(model_obj, model_path)
            except Exception as exc:
                logger.exception("Failed to save retrained model file (gRPC)")
                close_clearml_task(clearml_task)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Не удалось сохранить модель: {exc}")
                return ml_service_pb2.TrainModelResponse()

            # Логируем новую версию модели в ClearML
            clearml_model_id = log_model_to_clearml(
                task=clearml_task,
                model_path=str(model_path),
                model_name=model_row.name,
            )
            close_clearml_task(clearml_task)

            model_row.model_class = request.model_class
            model_row.hyperparams = hyperparams
            model_row.local_path = str(model_path)
            model_row.status = "trained"
            model_row.clearml_model_id = clearml_model_id
            db.commit()
            db.refresh(model_row)

            logger.info("gRPC model id=%s retrained successfully", model_row.id)

            return ml_service_pb2.TrainModelResponse(
                model_id=model_row.id,
                status=model_row.status,
            )
        finally:
            db.close()


    #DeleteModel
    def DeleteModel(self, request, context):
        logger.info("gRPC DeleteModel: model_id=%s", request.model_id)
        db = SessionLocal()
        try:
            model_row = (
                db.query(Model)
                .filter(Model.id == request.model_id)
                .first()
            )
            if model_row is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Model not found")
                return ml_service_pb2.DeleteModelResponse()

            model_row.status = "deleted"
            db.commit()

            try:
                if os.path.exists(model_row.local_path):
                    os.remove(model_row.local_path)
            except OSError:
                logger.exception("Failed to remove model file (gRPC)")

            return ml_service_pb2.DeleteModelResponse(status="deleted")
        finally:
            db.close()

    #Predict
    def Predict(self, request, context):
        logger.info(
            "gRPC Predict: model_id=%s, n_samples=%s",
            request.model_id,
            len(request.features),
        )
        db = SessionLocal()
        try:
            model_row = (
                db.query(Model)
                .filter(Model.id == request.model_id)
                .first()
            )
            if model_row is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Model not found")
                return ml_service_pb2.PredictResponse()

            if model_row.status != "trained":
                msg = f"Модель в статусе '{model_row.status}', инференс невозможен"
                logger.error(msg)
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(msg)
                return ml_service_pb2.PredictResponse()

            if not os.path.exists(model_row.local_path):
                logger.error("Model file not found: %s", model_row.local_path)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Файл модели не найден на диске")
                return ml_service_pb2.PredictResponse()

            try:
                model_obj = joblib.load(model_row.local_path)
            except Exception as exc:
                logger.exception("Failed to load model file (gRPC)")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Не удалось загрузить модель: {exc}")
                return ml_service_pb2.PredictResponse()

            features_list = []
            for fv in request.features:
                features_list.append(list(fv.values))

            X = np.array(features_list, dtype=float)

            try:
                preds = model_obj.predict(X)
            except Exception as exc:
                logger.exception("Model prediction failed (gRPC)")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Ошибка инференса: {exc}")
                return ml_service_pb2.PredictResponse()

            resp = ml_service_pb2.PredictResponse()
            for value in preds:
                resp.predictions.append(int(value))

            logger.info("gRPC predictions done: %s samples", len(resp.predictions))
            return resp
        finally:
            db.close()


# Функция запуска gRPC-сервера
def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(
        MLServiceServicer(),
        server,
    )

    port = os.getenv("GRPC_PORT", "50051")
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting gRPC server on %s", listen_addr)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

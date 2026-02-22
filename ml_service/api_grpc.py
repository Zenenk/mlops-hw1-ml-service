from __future__ import annotations

import json
from concurrent import futures
from typing import Iterator

import grpc
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .db_models import Base
from .grpc import ml_service_pb2, ml_service_pb2_grpc
from .logging_config import setup_logging
from .model_registry import UnsupportedModelError, get_available_model_classes
from .services import (
    DatasetNotFoundError,
    ModelBadStatusError,
    ModelNotFoundError,
    delete_dataset_service,
    delete_model_service,
    list_datasets_service,
    list_models_service,
    predict_model_service,
    retrain_model_service,
    train_model_service,
)

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

Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    """
    Реализация методов gRPC-сервиса, опирающаяся на ту же БД
    и тот же слой бизнес-логики, что и REST API.
    """

    def HealthCheck(self, request, context):
        logger.info("gRPC HealthCheck called")
        return ml_service_pb2.HealthResponse(
            status="ok",
            version="0.1.0",
        )

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

    def ListDatasets(self, request, context):
        logger.info("gRPC ListDatasets called")
        db = SessionLocal()
        try:
            items = list_datasets_service(db)
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

    def ListModels(self, request, context):
        logger.info("gRPC ListModels called")
        db = SessionLocal()
        try:
            items = list_models_service(db)
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

    def TrainModel(self, request, context):
        logger.info(
            "gRPC TrainModel: name=%s class=%s dataset_id=%s",
            request.name,
            request.model_class,
            request.dataset_id,
        )
        db = SessionLocal()
        try:
            try:
                hyperparams = json.loads(request.hyperparams_json) if request.hyperparams_json else {}
                if not isinstance(hyperparams, dict):
                    raise ValueError("JSON гиперпараметров должен быть объектом (словарём).")
            except Exception as exc:  # noqa: BLE001
                msg = f"Некорректный JSON гиперпараметров: {exc}"
                logger.error(msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return ml_service_pb2.TrainModelResponse()

            try:
                model_row = train_model_service(
                    db=db,
                    name=request.name,
                    model_class=request.model_class,
                    dataset_id=request.dataset_id,
                    hyperparams=hyperparams,
                )
            except DatasetNotFoundError as exc:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except UnsupportedModelError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except ValueError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except TypeError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except RuntimeError as exc:
                logger.exception("Model training failed (gRPC)")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()

            return ml_service_pb2.TrainModelResponse(
                model_id=model_row.id,
                status=model_row.status,
            )
        finally:
            db.close()

    def RetrainModel(self, request, context):
        logger.info("gRPC RetrainModel: model_id=%s", request.model_id)
        db = SessionLocal()
        try:
            try:
                hyperparams = json.loads(request.hyperparams_json) if request.hyperparams_json else {}
                if not isinstance(hyperparams, dict):
                    raise ValueError("JSON гиперпараметров должен быть объектом (словарём).")
            except Exception as exc:  # noqa: BLE001
                msg = f"Некорректный JSON гиперпараметров: {exc}"
                logger.error(msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(msg)
                return ml_service_pb2.TrainModelResponse()

            try:
                model_row = retrain_model_service(
                    db=db,
                    model_id=request.model_id,
                    model_class=request.model_class,
                    hyperparams=hyperparams,
                )
            except ModelNotFoundError as exc:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except DatasetNotFoundError as exc:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except UnsupportedModelError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except ValueError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except TypeError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except RuntimeError as exc:
                logger.exception("Model retraining failed (gRPC)")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()

            return ml_service_pb2.TrainModelResponse(
                model_id=model_row.id,
                status=model_row.status,
            )
        finally:
            db.close()

    def Predict(self, request, context):
        """
        Инференс через gRPC.

        Ожидается, что request.features — повторяющееся поле
        с сообщениями FeatureVector { repeated double values = 1; }.
        """
        logger.info("gRPC Predict: model_id=%s", request.model_id)
        db = SessionLocal()
        try:
            features_list = [list(fv.values) for fv in request.features]
            try:
                predictions = predict_model_service(
                    db=db,
                    model_id=request.model_id,
                    features=features_list,
                )
            except ModelNotFoundError as exc:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(exc))
                return ml_service_pb2.PredictResponse()
            except ModelBadStatusError as exc:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(str(exc))
                return ml_service_pb2.PredictResponse()
            except FileNotFoundError as exc:
                logger.exception("Model file not found in gRPC Predict")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.PredictResponse()
            except RuntimeError as exc:
                logger.exception("Prediction failed (gRPC)")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.PredictResponse()

            resp = ml_service_pb2.PredictResponse()
            resp.predictions.extend(predictions)
            return resp
        finally:
            db.close()

    def DeleteDataset(self, request, context):  # noqa: N802
        logger.info("gRPC DeleteDataset: id=%s", request.dataset_id)
        db = SessionLocal()
        try:
            try:
                delete_dataset_service(db, request.dataset_id)
            except DatasetNotFoundError as exc:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(exc))
                return ml_service_pb2.DeleteDatasetResponse()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to delete dataset via gRPC")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.DeleteDatasetResponse()

            return ml_service_pb2.DeleteDatasetResponse(status="deleted")
        finally:
            db.close()

    def DeleteModel(self, request, context):
        logger.info("gRPC DeleteModel: id=%s", request.model_id)
        db = SessionLocal()
        try:
            try:
                delete_model_service(db, request.model_id)
            except ModelNotFoundError as exc:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(exc))
                return ml_service_pb2.DeleteModelResponse()
            except Exception as exc:
                logger.exception("Failed to delete model via gRPC")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.DeleteModelResponse()

            return ml_service_pb2.DeleteModelResponse(status="deleted")
        finally:
            db.close()


def serve() -> None:
    """
    Точка входа для gRPC-сервера.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    logger.info("Starting gRPC server on :50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

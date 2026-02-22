from __future__ import annotations

import json
from concurrent import futures
from typing import Dict

import grpc
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings
from ..db_models import Base
from ..logging_config import setup_logging
from ..model_registry import UnsupportedModelError, get_available_model_classes
from ..services import (
    delete_model_service,
    list_datasets_service,
    list_models_service,
    predict_service,
    retrain_model_service,
    train_model_service,
)
from . import ml_service_pb2
from . import ml_service_pb2_grpc

logger = setup_logging(__name__)

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


def _parse_hyperparams_json(s: str) -> Dict:
    if not s:
        return {}
    data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("hyperparams_json должен быть JSON-объектом (словарём).")
    return data


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    def HealthCheck(self, request, context):
        logger.info("gRPC HealthCheck called")
        return ml_service_pb2.HealthResponse(status="ok", version="0.1.0")

    def ListModelClasses(self, request, context):
        logger.info("gRPC ListModelClasses called")
        registry = get_available_model_classes()
        resp = ml_service_pb2.ModelClassesResponse()
        for model_id, meta in registry.items():
            resp.classes.append(
                ml_service_pb2.ModelClassInfo(
                    id=model_id,
                    description=meta["description"],
                    default_hyperparams_json=json.dumps(meta.get("default_hyperparams") or {}),
                )
            )
        return resp

    def ListDatasets(self, request, context):
        logger.info("gRPC ListDatasets called")
        db = SessionLocal()
        try:
            items = list_datasets_service(db)
            resp = ml_service_pb2.DatasetListResponse()
            for d in items:
                resp.datasets.append(
                    ml_service_pb2.DatasetInfo(
                        id=int(d.id),
                        name=d.name,
                        path=d.path,
                        description=d.description or "",
                        version=d.version or "",
                    )
                )
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
                resp.models.append(
                    ml_service_pb2.ModelInfo(
                        id=int(m.id),
                        name=m.name,
                        model_class=m.model_class,
                        dataset_id=int(m.dataset_id),
                        status=m.status,
                        hyperparams_json=json.dumps(m.hyperparams or {}),
                        clearml_model_id=m.clearml_model_id or "",
                    )
                )
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
                hyperparams = _parse_hyperparams_json(request.hyperparams_json)
            except Exception as exc:  # noqa: BLE001
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Некорректный hyperparams_json: {exc}")
                return ml_service_pb2.TrainModelResponse()

            try:
                model_row = train_model_service(
                    db,
                    name=request.name,
                    model_class=request.model_class,
                    dataset_id=int(request.dataset_id),
                    hyperparams=hyperparams,
                )
            except UnsupportedModelError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except ValueError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except RuntimeError as exc:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()

            return ml_service_pb2.TrainModelResponse(model_id=int(model_row.id), status=model_row.status)
        finally:
            db.close()

    def RetrainModel(self, request, context):
        logger.info("gRPC RetrainModel: model_id=%s model_class=%s", request.model_id, request.model_class)
        db = SessionLocal()
        try:
            try:
                hyperparams = _parse_hyperparams_json(request.hyperparams_json)
            except Exception as exc:  # noqa: BLE001
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Некорректный hyperparams_json: {exc}")
                return ml_service_pb2.TrainModelResponse()

            model_class = request.model_class if request.model_class else None

            try:
                model_row = retrain_model_service(
                    db,
                    model_id=int(request.model_id),
                    model_class=model_class,
                    hyperparams=hyperparams,
                )
            except UnsupportedModelError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except ValueError as exc:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()
            except RuntimeError as exc:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.TrainModelResponse()

            return ml_service_pb2.TrainModelResponse(model_id=int(model_row.id), status=model_row.status)
        finally:
            db.close()

    def DeleteModel(self, request, context):
        logger.info("gRPC DeleteModel: model_id=%s", request.model_id)
        db = SessionLocal()
        try:
            try:
                delete_model_service(db, int(request.model_id))
            except ValueError as exc:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(exc))
                return ml_service_pb2.DeleteModelResponse(status="not_found")
            return ml_service_pb2.DeleteModelResponse(status="deleted")
        finally:
            db.close()

    def Predict(self, request, context):
        logger.info("gRPC Predict: model_id=%s", request.model_id)
        db = SessionLocal()
        try:
            features = [list(vec.values) for vec in request.features]
            try:
                preds = predict_service(db, model_id=int(request.model_id), features=features)
            except ValueError as exc:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return ml_service_pb2.PredictResponse()
            except RuntimeError as exc:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return ml_service_pb2.PredictResponse()

            resp = ml_service_pb2.PredictResponse()
            resp.predictions.extend([int(p) for p in preds])
            return resp
        finally:
            db.close()


def serve(port: int = 50051) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting gRPC server on port %s", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
from __future__ import annotations

import json

import grpc
from google.protobuf.empty_pb2 import Empty

from ml_service import ml_service_pb2, ml_service_pb2_grpc

GRPC_ADDR = "localhost:50051"


def main() -> None:
    with grpc.insecure_channel(GRPC_ADDR) as channel:
        stub = ml_service_pb2_grpc.MLServiceStub(channel)

        # ---------- HealthCheck ----------
        health = stub.HealthCheck(ml_service_pb2.HealthRequest())
        print(f"HealthCheck: {health.status} {health.version}")

        # ---------- Список классов моделей ----------
        classes_resp = stub.ListModelClasses(Empty())
        print("Model classes:")
        for cls in classes_resp.classes:
            print(
                f" - {cls.id} {cls.description}\n"
                f"   defaults: {cls.default_hyperparams_json}"
            )

        # ---------- Список датасетов ----------
        ds_resp = stub.ListDatasets(Empty())
        dataset_ids = [d.id for d in ds_resp.datasets]
        print(f"Datasets: {dataset_ids}")
        if not dataset_ids:
            print("Нет датасетов, gRPC-обучение пропущено.")
            return

        dataset_id = dataset_ids[0]

        # ---------- Обучение модели ----------
        hyperparams = {
            "C": 1.0,
            "max_iter": 200,
            "solver": "lbfgs",
            "random_state": 42,
        }
        train_req = ml_service_pb2.TrainModelRequest(
            name="grpc_lr_iris",
            model_class="logistic_regression",
            dataset_id=dataset_id,
            hyperparams_json=json.dumps(hyperparams),
        )

        try:
            train_resp = stub.TrainModel(train_req)
            print(
                f"TrainModel: model_id = {train_resp.model_id} "
                f"status = {train_resp.status}"
            )
        except grpc.RpcError as e:
            print(
                f"TrainModel RPC failed: code={e.code()} "
                f"details={e.details()}"
            )
            return

        model_id = train_resp.model_id

        #Список моделей
        models_resp = stub.ListModels(Empty())
        model_ids = [m.id for m in models_resp.models]
        print(f"Models: {model_ids}")

        #Инференс
        predict_req = ml_service_pb2.PredictRequest(model_id=model_id)

        fv1 = predict_req.features.add()
        fv1.values.extend([5.1, 3.5, 1.4, 0.2])

        fv2 = predict_req.features.add()
        fv2.values.extend([6.2, 3.4, 5.4, 2.3])

        try:
            predict_resp = stub.Predict(predict_req)
            print(f"Predictions: {list(predict_resp.predictions)}")
        except grpc.RpcError as e:
            print(
                f"Predict RPC failed: code={e.code()} "
                f"details={e.details()}"
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

import grpc

from ml_service.grpc import ml_service_pb2, ml_service_pb2_grpc


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = ml_service_pb2_grpc.MLServiceStub(channel)

    print("== HealthCheck ==")
    health = stub.HealthCheck(ml_service_pb2.HealthRequest())
    print(health)

    print("\n== ListModelClasses ==")
    classes = stub.ListModelClasses(ml_service_pb2.Empty())
    for c in classes.classes:
        print("-", c.id, c.description, c.default_hyperparams_json)

    print("\n== ListDatasets ==")
    ds = stub.ListDatasets(ml_service_pb2.Empty())
    for d in ds.datasets:
        print("-", d.id, d.name, d.path)

    if not ds.datasets:
        print("\nNo datasets found. Upload dataset via REST first.")
        return

    dataset_id = ds.datasets[0].id

    print("\n== TrainModel ==")
    hyper = {"C": 0.7, "max_iter": 300, "solver": "lbfgs", "random_state": 42}
    train = stub.TrainModel(
        ml_service_pb2.TrainModelRequest(
            name="grpc_lr_model",
            model_class="logistic_regression",
            dataset_id=dataset_id,
            hyperparams_json=json.dumps(hyper),
        )
    )
    print(train)

    print("\n== ListModels ==")
    models = stub.ListModels(ml_service_pb2.Empty())
    for m in models.models:
        print("-", m.id, m.name, m.model_class, m.status, m.hyperparams_json)

    print("\n== Predict ==")
    req = ml_service_pb2.PredictRequest(
        model_id=train.model_id,
        features=[
            ml_service_pb2.FloatVector(values=[5.1, 3.5, 1.4, 0.2]),
            ml_service_pb2.FloatVector(values=[6.2, 3.4, 5.4, 2.3]),
        ],
    )
    pred = stub.Predict(req)
    print(pred)

    print("\n== DeleteModel ==")
    deleted = stub.DeleteModel(ml_service_pb2.DeleteModelRequest(model_id=train.model_id))
    print(deleted)


if __name__ == "__main__":
    main()
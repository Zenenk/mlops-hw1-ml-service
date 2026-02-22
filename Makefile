MINIKUBE_PROFILE = mlops-hw1
IMAGE_NAME = mlops-hw1-app
IMAGE_TAG = latest

.PHONY: help
help:
	@echo "Доступные цели:"
	@echo "  make minikube-start   - запустить minikube (driver=docker)"
	@echo "  make minikube-stop    - остановить minikube"
	@echo "  make minikube-delete  - удалить minikube кластер"
	@echo "  make docker-build     - собрать образ приложения на хосте + загрузить в minikube"
	@echo "  make k8s-apply        - применить манифесты k8s"
	@echo "  make k8s-delete       - удалить ресурсы k8s"
	@echo "  make deploy-minikube  - полный цикл: start -> build -> apply"
	@echo "  make run-rest         - локальный запуск REST"
	@echo "  make run-grpc         - локальный запуск gRPC"
	@echo "  make run-dashboard    - локальный запуск dashboard"
	@echo "  make gen-grpc         - сгенерировать gRPC stubs + пропатчить импорты"

minikube-start:
	minikube start --profile $(MINIKUBE_PROFILE) --driver=docker --wait=all

minikube-stop:
	minikube stop --profile $(MINIKUBE_PROFILE)

minikube-delete:
	minikube delete --profile $(MINIKUBE_PROFILE)

# Сборка на ХОСТЕ, затем загрузка образа в minikube.
# Это обходит проблемы DNS/доступа к Docker Hub внутри minikube docker-env.
docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .
	minikube -p $(MINIKUBE_PROFILE) image load $(IMAGE_NAME):$(IMAGE_TAG)

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -n mlops-hw1 -f k8s/backend-pvc.yaml
	kubectl apply -n mlops-hw1 -f k8s/minio.yaml
	kubectl apply -n mlops-hw1 -f k8s/backend-deployment.yaml
	kubectl apply -n mlops-hw1 -f k8s/grpc-deployment.yaml
	kubectl apply -n mlops-hw1 -f k8s/dashboard-deployment.yaml

k8s-delete:
	-kubectl delete -n mlops-hw1 -f k8s/dashboard-deployment.yaml
	-kubectl delete -n mlops-hw1 -f k8s/grpc-deployment.yaml
	-kubectl delete -n mlops-hw1 -f k8s/backend-deployment.yaml
	-kubectl delete -n mlops-hw1 -f k8s/minio.yaml
	-kubectl delete -n mlops-hw1 -f k8s/backend-pvc.yaml
	-kubectl delete -f k8s/namespace.yaml

deploy-minikube: minikube-start docker-build k8s-apply

.PHONY: run-rest
run-rest:
	poetry run python3 -m scripts.run_service rest

.PHONY: run-grpc
run-grpc:
	poetry run python3 -m scripts.run_service grpc

.PHONY: run-dashboard
run-dashboard:
	poetry run python3 -m scripts.run_service dashboard

.PHONY: gen-grpc
gen-grpc:
	poetry run python3 -m grpc_tools.protoc \
	  -I proto \
	  --python_out=ml_service/grpc \
	  --grpc_python_out=ml_service/grpc \
	  proto/ml_service.proto
	poetry run python3 scripts/patch_grpc_imports.py
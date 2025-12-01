MINIKUBE_PROFILE = mlops-hw1

.PHONY: help
help:
	@echo "Доступные цели:"
	@echo "  make minikube-start         - запустить minikube (driver=docker)"
	@echo "  make minikube-stop          - остановить minikube"
	@echo "  make minikube-delete        - удалить кластер minikube"
	@echo "  make docker-build-backend   - собрать Docker-образ backend внутри minikube"
	@echo "  make docker-build-dashboard - собрать Docker-образ dashboard внутри minikube"
	@echo "  make k8s-apply              - применить манифесты k8s (namespace, minio, backend, dashboard)"
	@echo "  make k8s-delete             - удалить ресурсы k8s"
	@echo "  make deploy-minikube        - полный цикл: start -> build -> apply"

minikube-start:
	minikube start --profile $(MINIKUBE_PROFILE) --driver=docker

minikube-stop:
	minikube stop --profile $(MINIKUBE_PROFILE)

minikube-delete:
	minikube delete --profile $(MINIKUBE_PROFILE)

docker-build-backend:
	@eval $$(minikube -p $(MINIKUBE_PROFILE) docker-env) && \
	docker build -t mlops-hw1-backend -f Dockerfile.backend .

docker-build-dashboard:
	@eval $$(minikube -p $(MINIKUBE_PROFILE) docker-env) && \
	docker build -t mlops-hw1-dashboard -f Dockerfile.dashboard .

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -n mlops-hw1 -f k8s/minio.yaml
	kubectl apply -n mlops-hw1 -f k8s/backend-deployment.yaml
	kubectl apply -n mlops-hw1 -f k8s/dashboard-deployment.yaml

k8s-delete:
	-kubectl delete -n mlops-hw1 -f k8s/dashboard-deployment.yaml
	-kubectl delete -n mlops-hw1 -f k8s/backend-deployment.yaml
	-kubectl delete -n mlops-hw1 -f k8s/minio.yaml
	-kubectl delete -f k8s/namespace.yaml

deploy-minikube: minikube-start docker-build-backend docker-build-dashboard k8s-apply

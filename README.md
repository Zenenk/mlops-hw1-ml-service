# ML Homework 1 – ML-сервис с REST, gRPC, DVC, Streamlit и Minikube

Проект реализует сервис для обучения и инференса простых ML-моделей
с REST API, gRPC API, интерактивным дашбордом на Streamlit,
версонированием датасетов через DVC и (опционально) логированием
экспериментов в ClearML с хранением весов на S3 (Minio).

1) REST API:
  1.1) обучение моделей с настраиваемыми гиперпараметрами;
  1.2) два класса моделей (`logistic_regression`, `random_forest`);
  1.3) список доступных классов моделей;
  1.4) инференс по конкретной модели;
  1.5) переобучение и логическое удаление моделей;
  1.6) эндпоинт статуса сервиса;
  1.7) эндпоинт для работы с датасетами (загрузка, список, удаление).
2) gRPC API:
  2.1) healthcheck, список классов, датасеты, модели, обучение, переобучение, инференс, удаление;
  2.2) отдельный скрипт клиента.
3) Streamlit-дашборд:
  3.1) вкладка «Статус» – проверка `/health`;
  3.2) вкладка «Датасеты» – загрузка/просмотр/удаление датасетов;
  3.3) вкладка «Обучение» – выбор датасета, класса модели и гиперпараметров (JSON);
  3.4) вкладка «Инференс» – выбор модели и признаков (JSON).
4) Инфраструктура:
  4.1) сервис + дашборд запускаются в Minikube;
  4.2) Minio для S3-хранилища;
  4.3) DVC для версионирования датасетов и кэша;
  4.4) ClearML-сервер для экспериментов и моделей;
  4.5) Makefile для удобного запуска.

---

## 1. Стек и структура проекта

### Технологии

- Python 3.10
- FastAPI + Uvicorn (REST API)
- gRPC (`grpcio`, `grpcio-tools`)
- scikit-learn (ML-модели)
- SQLAlchemy + SQLite
- Streamlit
- DVC
- Minio
- ClearML
- Docker, Minikube, kubectl
- Poetry
- ruff, isort, mypy

### Структура каталогов

```text
.
├── Dockerfile.backend
├── Dockerfile.dashboard
├── Makefile
├── README.md
├── clearml
│   └── docker-compose.yml
├── clearml.conf
├── dashboard
│   ├── __init__.py
│   └── app.py
├── data
│   ├── datasets
│   │   ├── iris_tiny.csv
│   │   └── iris_tiny.csv.dvc
│   └── models
│       └── grpc_logistic_regression_1.joblib
├── k8s
│   ├── backend-deployment.yaml
│   ├── backend.yaml
│   ├── dashboard-deployment.yaml
│   ├── minio.yaml
│   └── namespace.yaml
├── ml_service
│   ├── __init__.py
│   ├── api_grpc.py
│   ├── api_rest.py
│   ├── clearml_utils.py
│   ├── config.py
│   ├── db_models.py
│   ├── logging_config.py
│   ├── ml_service_pb2.py
│   ├── ml_service_pb2_grpc.py
│   ├── model_registry.py
│   └── schemas.py
├── ml_service.db
├── poetry.lock
├── proto
│   └── ml_service.proto
├── pyproject.toml
└── scripts
    ├── __pycache__
    │   └── grpc_client.cpython-310.pyc
    └── grpc_client.py
```

2.Установка и запуск локально
2.1. Предварительные требования

- Python 3.10

- Poetry

- Docker

- Браузер для работы с дашбордом

2.2.Установка зависимостей

```text

git clone <URL_РЕПОЗИТОРИЯ> mlops-hw1-ml-service
cd mlops-hw1-ml-service

```

```text

poetry install
source "$(poetry env info --path)/bin/activate"

```

Зависимости проекта описаны в pyproject.toml, конкретные версии зафиксированы в poetry.lock.

2.3. Запуск REST-сервиса

```text

cd ~/mlops-hw1-ml-service
source "$(poetry env info --path)/bin/activate"

export CLEARML_ENABLED=false  # ClearML по умолчанию отключён

uvicorn ml_service.api_rest:app --host 0.0.0.0 --port 8000 --reload

```

Проверки:

Health-check: <http://localhost:8000/health>

Swagger UI: <http://localhost:8000/docs>

ReDoc: <http://localhost:8000/redoc>

2.4. Запуск gRPC-сервера

В отдельном терминале:

```text

cd ~/mlops-hw1-ml-service
source "$(poetry env info --path)/bin/activate"

python -m ml_service.api_grpc

```

Проверка через клиент:

```text

cd ~/mlops-hw1-ml-service
source "$(poetry env info --path)/bin/activate"

python -m scripts.grpc_client

```

Скрипт делает HealthCheck, выводит список классов моделей и датасетов,
обучает модель и выполняет инференс на двух объектах.

2.5. Запуск Streamlit-дашборда

Ещё один терминал:

```text

cd ~/mlops-hw1-ml-service
source "$(poetry env info --path)/bin/activate"

export CLEARML_ENABLED=false
streamlit run dashboard/app.py

```

По умолчанию дашборд будет доступен по адресу: <http://localhost:8501>
.

В левой колонке в поле Backend URL должно быть <http://localhost:8000>.

3.Запуск в Minikube (backend + dashboard + Minio)
3.1. Предварительные требования

Docker (драйвер docker)

minikube

kubectl

3.2. Запуск кластера

```text

minikube start -p mlops-hw1 --driver=docker

```

3.3. Сборка Docker-образов внутри кластера

```text

eval "$(minikube -p mlops-hw1 docker-env)"

cd ~/mlops-hw1-ml-service

docker build -t mlops-hw1-backend   -f Dockerfile.backend   .
docker build -t mlops-hw1-dashboard -f Dockerfile.dashboard .

```

3.4. Применение манифестов

```text

cd ~/mlops-hw1-ml-service

kubectl apply -f k8s/namespace.yaml
kubectl apply -n mlops-hw1 -f k8s/minio.yaml
kubectl apply -n mlops-hw1 -f k8s/backend-deployment.yaml
kubectl apply -n mlops-hw1 -f k8s/dashboard-deployment.yaml

```

Проверка:

```text

kubectl get pods -n mlops-hw1
kubectl get svc  -n mlops-hw1

```

3.5. Доступ к дашборду

```text

minikube -p mlops-hw1 service mlops-dashboard -n mlops-hw1 --url

```

Команда вернёт URL вида <http://127.0.0.1:30xxx> — открыть его в браузере.

3.6. Доступ к Minio

```text

kubectl port-forward svc/minio -n mlops-hw1 9000:9000 9001:9001

```

После этого:

API Minio: <http://localhost:9000>

Web-интерфейс Minio: <http://localhost:9001>

Учётные данные и настройки бакета смотреть в k8s/minio.yaml.

4.DVC и датасеты

DVC уже инициализирован, пример датасета: data/datasets/iris_tiny.csv.

DVC-файл: data/datasets/iris_tiny.csv.dvc.

Добавление нового датасета:

```text

dvc add data/datasets/my_dataset.csv
git add data/datasets/my_dataset.csv.dvc

```

При использовании удалённого хранилища (S3/Minio) нужно настроить
remote в .dvc/config, после чего можно выполнять:

```text

dvc push

```

5.Работа с REST API (примеры запросов)

Все примеры ниже предполагают, что REST-сервис запущен
на <http://localhost:8000>.

5.1. Health-check

```text

curl <http://localhost:8000/health>

```

5.2. Список доступных классов моделей

```text

curl <http://localhost:8000/model_classes>

```

5.3. Загрузка датасета

```text

curl -X POST "<http://localhost:8000/datasets>" \
  -F "file=@data/datasets/iris_tiny.csv"

```

5.4. Список датасетов

```text

curl <http://localhost:8000/datasets>

```

5.5. Обучение модели

```text

curl -X POST "<http://localhost:8000/models/train>" \
  -H "Content-Type: application/json" \
  -d '{
        "name": "lr_iris_rest",
        "model_class": "logistic_regression",
        "dataset_id": 1,
        "hyperparams": {
          "C": 1.0,
          "max_iter": 200,
          "solver": "lbfgs",
          "random_state": 42
        }
      }'

```

5.6. Список моделей

```text

curl <http://localhost:8000/models>

```

5.7. Инференс

```text

curl -X POST "<http://localhost:8000/models/1/predict>" \
  -H "Content-Type: application/json" \
  -d '{
        "features": [
          [5.1, 3.5, 1.4, 0.2],
          [6.2, 3.4, 5.4, 2.3]
        ]
      }'

```

Число признаков в каждом векторе должно совпадать
с числом признаков в датасете, на котором обучалась модель.

5.8. Переобучение модели

```text

curl -X POST "<http://localhost:8000/models/1/retrain>" \
  -H "Content-Type: application/json" \
  -d '{
        "model_class": "random_forest",
        "hyperparams": {
          "n_estimators": 200,
          "max_depth": 3,
          "n_jobs": -1,
          "random_state": 42
        }
      }'

```

5.9. Логическое удаление модели

```text

curl -X DELETE "<http://localhost:8000/models/1>"

```

6.Проверка стиля и качества кода

Для проверки используются:

```text

ruff check .
isort .

```

7.ClearML (опционально, доп. баллы)

Интеграция с ClearML отключена по умолчанию
переменной окружения

```text

CLEARML_ENABLED=false.

```

7.1. Запуск ClearML Server

```text

cd clearml
docker compose up -d    # либо docker-compose up -d

```

Порты по умолчанию (уточнить в clearml/docker-compose.yml):

Web-интерфейс: <http://localhost:8080>

API-сервер: <http://localhost:8008>

File-server: <http://localhost:8081>

7.2. Получение API-ключей

Открыть <http://localhost:8080> в браузере.

Зарегистрироваться / войти.

В правом верхнем углу: Settings → Workspace → Create new credentials.

Скопировать API Access Key и API Secret Key.

7.3. Настройка clearml.conf

Открыть файл clearml.conf в корне проекта
(или создать ~/.clearml.conf) и прописать ключи:

api {
  web_server: "<http://localhost:8080>"
  api_server: "<http://localhost:8008>"
  files_server: "<http://localhost:8081>"
  credentials {
    api_key: "ВАШ_API_ACCESS_KEY"
    secret_key: "ВАШ_API_SECRET_KEY"
  }
}

!!!Важно: не коммитить реальные ключи в публичный репозиторий.

7.4. Запуск сервисов с ClearML

Перед запуском REST/gRPC-сервисов включить ClearML:

```text

export CLEARML_ENABLED=true

```

Далее:

```text

# REST

uvicorn ml_service.api_rest:app --host 0.0.0.0 --port 8000 --reload

# gRPC

python -m ml_service.api_grpc

```

7.5. Проверка интеграции

Через дашборд или curl запустить обучение модели.

В веб-интерфейсе ClearML (<http://localhost:8080>)
проверить, что появился новый эксперимент.

В разделе Models увидеть сохранённую модель
(при корректной настройке S3/Minio веса будут в хранилище).

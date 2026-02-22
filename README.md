# ML Homework 1 – ML-сервис с REST, gRPC, DVC, Streamlit и Minikube

Проект реализует сервис для обучения и инференса простых ML-моделей

Реализовано:
REST API (FastAPI) для загрузки датасетов, обучения моделей, инференса, переобучения и удаления
gRPC API с аналогичной функциональностью и отдельным клиентом для проверки
Интерактивный дашборд (Streamlit), который вызывает REST-эндпоинты сервиса
DVC для версионирования датасетов и пуша артефактов в S3-совместимое хранилище Minio
Запуск в Minikube через Makefile

Интеграция ClearML — опциональная. Если включена, обучение логируется как эксперимент, а модель может быть опубликована в ClearML/Minio.

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
- ruff

### Структура каталогов

```text
.
├── Dockerfile
├── Dockerfile.backend
├── Dockerfile.dashboard
├── Makefile
├── README.md
├── alembic
│   ├── README
│   ├── __pycache__
│   │   └── env.cpython-310.pyc
│   ├── env.py
│   ├── script.py.mako
│   └── versions
├── alembic.ini
├── clearml
│   └── docker-compose.yml
├── clearml.conf
├── dashboard
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   └── app.cpython-310.pyc
│   └── app.py
├── data
│   ├── datasets
│   └── models
├── k8s
│   ├── backend-deployment.yaml
│   ├── backend-pvc.yaml
│   ├── dashboard-deployment.yaml
│   ├── grpc-deployment.yaml
│   ├── minio.yaml
│   └── namespace.yaml
├── ml_service
│   ├── __init__.py
│   ├── api_grpc.py
│   ├── api_rest.py
│   ├── clearml_utils.py
│   ├── config.py
│   ├── db_models.py
│   ├── dvc_utils.py
│   ├── grpc
│   │   ├── ml_service_pb2.py
│   │   ├── ml_service_pb2_grpc.py
│   │   └── server.py
│   ├── logging_config.py
│   ├── model_registry.py
│   ├── schemas.py
│   └── services.py
├── ml_service.db
├── poetry.lock
├── proto
│   └── ml_service.proto
├── pyproject.toml
└── scripts
    ├── __init__.py
    ├── grpc_client.py
    ├── patch_grpc_imports.py
    └── run_service.py
```

2.Установка и запуск локально
2.1. Предварительные требования

- Python 3.10

- Poetry

- Docker

- Minikube

- kubectl

- Браузер для работы с дашбордом

2.2.Установка зависимостей

```text

git clone <URL_РЕПОЗИТОРИЯ> mlops-hw1-ml-service
cd mlops-hw1-ml-service

```

```text

poetry install
poetry run alembic upgrade head
poetry run ruff check .

```

Зависимости проекта описаны в pyproject.toml, конкретные версии зафиксированы в poetry.lock.

2.3. Запуск REST-сервиса

```text

cd ~/mlops-hw1-ml-service
poetry run python3 -m scripts.run_service rest

```

Проверки:

Health-check: <http://localhost:8000/health>

Swagger UI: <http://localhost:8000/docs>

ReDoc: <http://localhost:8000/redoc>

Список классов моделей: <http://localhost:8000/model-classes>

Список датасетов: <http://localhost:8000/datasets>

2.4. Запуск gRPC-сервера

В отдельном терминале:

```text

cd ~/mlops-hw1-ml-service
poetry run python3 -m scripts.run_service grpc

```

Проверка через клиент:

```text

cd ~/mlops-hw1-ml-service
poetry run python3 -m scripts.grpc_client

```

Скрипт делает HealthCheck, выводит список классов моделей и датасетов,
обучает модель и выполняет инференс на двух объектах.

2.5. Запуск Streamlit-дашборда

Ещё один терминал:

```text

cd ~/mlops-hw1-ml-service
poetry run python3 -m scripts.run_service dashboard

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

cd ~/mlops-hw1-ml-service
make deploy-minikube

```

Проверка:

```text

kubectl get pods -n mlops-hw1
kubectl get svc -n mlops-hw1

```
3.3. Доступ к дашборду

```text

minikube -p mlops-hw1 service mlops-dashboard -n mlops-hw1 --url

```

Команда вернёт URL вида <http://127.0.0.1:xxxxx> — открыть его в браузере.

Проверка через port-forward:

```text

kubectl -n mlops-hw1 port-forward svc/mlops-backend 18000:8000

```

Healthcheck: <http://localhost:18000/health>

Пример загрузки датасета

```text

curl -sS -X POST "http://localhost:18000/datasets?description=iris_k8s" -F "file=@data/datasets/iris_tiny.csv"

```

path должен начинаться с /app/data/

3.4.gRPC


```text
kubectl -n mlops-hw1 port-forward svc/mlops-grpc 50051:50051

```

Проверка:

```text

cd ~/mlops-hw1-ml-service
poetry run python3 -m scripts.grpc_client

```

3.5. Доступ к Minio

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

DVC push:

```text

kubectl -n mlops-hw1 exec deploy/mlops-backend -- sh -lc 'echo DVC_ENABLED=$DVC_ENABLED; dvc remote list; dvc push -v'

```

Ожидаемо: “Everything is up to date.” или лог загрузки в s3://dvc/files/md5, в Minio смотреть bucket dvc путь files/md5/

5.Работа с REST API (примеры запросов)

Все примеры ниже предполагают, что REST-сервис запущен
на <http://localhost:8000>.

5.1. Health-check

```text

curl http://localhost:8000/health

```

5.2. Список доступных классов моделей

```text

curl http://localhost:8000/model_classes

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

```

7.ClearML

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

poetry run python3 -m scripts.run_service rest

# gRPC

poetry run python3 -m scripts.run_service rest

```

7.5. Проверка интеграции

Через дашборд или curl запустить обучение модели.

В веб-интерфейсе ClearML (<http://localhost:8080>)
проверить, что появился новый эксперимент.

В разделе Models увидеть сохранённую модель
(при корректной настройке S3/Minio веса будут в хранилище).

8. Команды Makefile

make minikube-start — запуск minikube

make docker-build — сборка образа

make k8s-apply — применить манифесты k8s

make k8s-delete — удалить ресурсы k8s

make deploy-minikube — полный цикл start - build - apply

make gen-grpc — генерация stubs и безопасный патч импортов

make run-rest — локальный REST

make run-grpc — локальный gRPC

make run-dashboard — локальный dashboard

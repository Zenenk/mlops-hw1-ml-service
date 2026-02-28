# API

## Запуск (локально)
Из корня репозитория:

- Установка зависимостей:
  - `poetry install`

- Запуск REST сервиса (как в проекте принято):
  - `poetry run python3 -m ml_service.api_rest`

- Запуск dashboard:
  - `poetry run python3 -m dashboard.app`

## Базовый URL
- REST: http://127.0.0.1:8000 (если в проекте другой порт — см. README.md)
- gRPC: см. README.md / ml_service/api_grpc.py

## REST эндпоинты
Этот файл должен быть обновлён агентом после реализации фичи и сверен с фактическими роутами в ml_service/api_rest.py.

## Примеры
Примеры curl должны быть добавлены агентом после реализации batch predict.
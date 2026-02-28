# PLAN_BATCH_PREDICT.md

## Цель
Добавить в ML-сервис batch-инференс по CSV-файлу:
- новый REST endpoint принимает CSV и возвращает предсказания для каждой строки
- dashboard получает возможность загрузить CSV и показать результаты

## Область изменений
- REST API: ml_service/api_rest.py
- Схемы: ml_service/schemas.py
- Бизнес-логика: ml_service/services.py (или новый модуль, если удобнее)
- Dashboard: dashboard/app.py
- Тесты: добавить/расширить pytest-тесты (если в проекте уже есть tests/, использовать её; иначе создать tests/)

## Требования к API
### Endpoint
POST /api/v1/models/{model_id}/predict-batch

### Вход
- multipart/form-data
- файл `file` (CSV)
- опционально: `has_header` (bool, default true)

### Выход
JSON:
- model_id
- rows: число обработанных строк
- predictions: список предсказаний (по порядку строк)
- optional: probabilities (если модель их поддерживает)

## План работ (Plan & Act)
1) Прочитать текущие эндпоинты предикта и обучение в api_rest.py и services.py.
2) Реализовать парсинг CSV (через стандартный csv модуль или pandas, если уже используется).
3) Добавить сервисную функцию batch_predict(model_id, rows) и переиспользовать существующий predict-пайплайн.
4) Добавить REST handler:
   - валидация файла
   - лимит на размер/число строк (разумный, чтобы не убить сервис)
   - обработка ошибок
5) Добавить тесты:
   - успешный CSV (несколько строк)
   - пустой файл
   - неверный формат/колонки
6) Обновить dashboard:
   - форма загрузки CSV
   - отображение predictions таблицей
7) Запустить линтеры и тесты:
   - poetry run ruff format .
   - poetry run ruff check .
   - poetry run pytest -q
8) Обновить документацию:
   - API.md: новый endpoint + пример curl
   - CHANGELOG.md: запись о batch predict
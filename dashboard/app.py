from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

BACKEND_URL_DEFAULT = "http://localhost:8000"


def get_backend_url() -> str:
    """
    Определяет URL backend-а.

    Сначала смотрим переменную окружения BACKEND_URL,
    затем значение, введённое пользователем в сайдбаре,
    потом дефолт.
    """
    env_url = os.getenv("BACKEND_URL", BACKEND_URL_DEFAULT)
    # Используем session_state, чтобы не терять URL между вкладками
    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = env_url

    return st.session_state["backend_url"]


def set_backend_url(new_url: str) -> None:
    st.session_state["backend_url"] = new_url



# Вспомогательные функции для REST-вызовов
def api_get(path: str) -> Optional[Dict[str, Any]]:
    url = get_backend_url().rstrip("/") + path
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        st.error(f"Ошибка запроса GET {url}: {exc}")
        return None

    if not resp.ok:
        st.error(f"Ошибка GET {url}: {resp.status_code} {resp.text}")
        return None

    try:
        return resp.json()
    except ValueError:
        st.error(f"Не удалось распарсить JSON от {url}")
        return None


def api_post_json(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = get_backend_url().rstrip("/") + path
    try:
        resp = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as exc:
        st.error(f"Ошибка запроса POST {url}: {exc}")
        return None

    if not resp.ok:
        st.error(f"Ошибка POST {url}: {resp.status_code} {resp.text}")
        return None

    try:
        return resp.json()
    except ValueError:
        st.error(f"Не удалось распарсить JSON от {url}")
        return None


def api_post_file(
    path: str,
    file_name: str,
    file_bytes: bytes,
    description: str,
) -> Optional[Dict[str, Any]]:
    """
    Загружаем файл датасета через /datasets.

    В FastAPI эндпоинт ожидает:
    - file: UploadFile
    - description: str (query-param)
    """
    base_url = get_backend_url().rstrip("/")
    url = base_url + path
    files = {"file": (file_name, file_bytes)}
    params = {"description": description}
    try:
        resp = requests.post(url, files=files, params=params, timeout=60)
    except requests.RequestException as exc:
        st.error(f"Ошибка загрузки файла на {url}: {exc}")
        return None

    if not resp.ok:
        st.error(f"Ошибка загрузки файла {file_name}: {resp.status_code} {resp.text}")
        return None

    try:
        return resp.json()
    except ValueError:
        st.error("Не удалось распарсить JSON при загрузке файла датасета")
        return None


def api_delete(path: str) -> bool:
    url = get_backend_url().rstrip("/") + path
    try:
        resp = requests.delete(url, timeout=10)
    except requests.RequestException as exc:
        st.error(f"Ошибка DELETE {url}: {exc}")
        return False

    if not resp.ok:
        st.error(f"Ошибка DELETE {url}: {resp.status_code} {resp.text}")
        return False

    return True



# Вкладка: статус сервиса
def render_status_tab() -> None:
    st.header("Статус сервиса")

    health = api_get("/health")
    if health is None:
        st.error("Не удалось получить статус сервиса.")
        return

    st.success(f"Сервис работает. Статус: {health.get('status')}, версия: {health.get('version')}")



# Вкладка: датасеты
def render_datasets_tab() -> None:
    st.header("Управление датасетами")

    st.subheader("Загрузить новый датасет")
    uploaded_file = st.file_uploader(
        "Выберите CSV или JSON файл с колонкой 'target'",
        type=["csv", "json"],
    )
    description = st.text_input("Описание датасета (опционально)", "")

    if uploaded_file is not None and st.button("Загрузить датасет"):
        content = uploaded_file.read()
        result = api_post_file(
            path="/datasets",
            file_name=uploaded_file.name,
            file_bytes=content,
            description=description,
        )
        if result is not None:
            st.success(f"Датасет загружен. ID: {result.get('id')}, имя: {result.get('name')}")

    st.subheader("Список датасетов")

    datasets_resp = api_get("/datasets")
    if not datasets_resp:
        st.info("Датасетов пока нет.")
        return

    datasets = datasets_resp.get("datasets", [])
    if not datasets:
        st.info("Датасетов пока нет.")
        return

    for ds in datasets:
        with st.expander(f"Dataset #{ds['id']}: {ds['name']}"):
            st.write(f"ID: {ds['id']}")
            st.write(f"Путь: `{ds['path']}`")
            st.write(f"Описание: {ds.get('description') or '-'}")
            st.write(f"Версия (DVC): {ds.get('version') or '-'}")

            if st.button(f"Удалить датасет {ds['id']}", key=f"delete_ds_{ds['id']}"):
                ok = api_delete(f"/datasets/{ds['id']}")
                if ok:
                    st.success(f"Датасет {ds['id']} удалён.")
                    st.experimental_rerun()


# Вкладка: обучение моделей
def render_training_tab() -> None:
    st.header("Обучение моделей")

    classes_resp = api_get("/model-classes")
    if not classes_resp:
        st.error("Не удалось получить список классов моделей.")
        return

    model_classes: List[Dict[str, Any]] = classes_resp.get("classes", [])
    if not model_classes:
        st.error("Нет доступных классов моделей.")
        return

    datasets_resp = api_get("/datasets")
    datasets: List[Dict[str, Any]] = datasets_resp.get("datasets", []) if datasets_resp else []
    if not datasets:
        st.warning("Сначала загрузите хотя бы один датасет во вкладке 'Датасеты'.")
        return

    st.subheader("Параметры обучения")

    dataset_options = {f"#{d['id']}: {d['name']}": d["id"] for d in datasets}
    dataset_label = st.selectbox("Датасет", list(dataset_options.keys()))
    dataset_id = dataset_options[dataset_label]

    model_class_ids = [mc["id"] for mc in model_classes]
    model_class = st.selectbox("Класс модели", model_class_ids)

    selected_meta = next((mc for mc in model_classes if mc["id"] == model_class), None)
    if selected_meta:
        st.markdown(f"**Описание:** {selected_meta['description']}")
        default_hyperparams = selected_meta.get("default_hyperparams", {})
    else:
        default_hyperparams = {}

    st.markdown("**Гиперпараметры (JSON)**")
    default_json = json.dumps(default_hyperparams, indent=2, ensure_ascii=False)
    hyperparams_json = st.text_area(
        "Отредактируйте при необходимости",
        value=default_json,
        height=200,
    )

    model_name = st.text_input("Имя модели", f"{model_class}_model")

    if st.button("Обучить модель"):
        try:
            hyperparams = json.loads(hyperparams_json) if hyperparams_json.strip() else {}
            if not isinstance(hyperparams, dict):
                raise ValueError("JSON гиперпараметров должен быть объектом (словарём).")
        except Exception as exc:
            st.error(f"Ошибка парсинга гиперпараметров: {exc}")
            return

        payload = {
            "name": model_name,
            "model_class": model_class,
            "dataset_id": dataset_id,
            "hyperparams": hyperparams,
        }

        result = api_post_json("/models/train", payload)
        if result is not None:
            st.success(
                f"Модель обучена. ID: {result.get('model_id')}, статус: {result.get('status')}"
            )



# Вкладка: инференс
def render_inference_tab() -> None:
    st.header("Инференс")

    models_resp = api_get("/models")
    if not models_resp:
        st.error("Не удалось получить список моделей.")
        return

    models: List[Dict[str, Any]] = models_resp.get("models", [])
    if not models:
        st.info("Моделей пока нет. Сначала обучите модель во вкладке 'Обучение'.")
        return

    model_options = {
        f"#{m['id']}: {m['name']} ({m['model_class']}, status={m['status']})": m["id"]
        for m in models
    }
    model_label = st.selectbox("Модель", list(model_options.keys()))
    model_id = model_options[model_label]

    st.markdown(
        """
        Введите признаки в формате JSON — список объектов,
        где каждый объект — список чисел (float):

        ```json
        [
          [5.1, 3.5, 1.4, 0.2],
          [6.2, 3.4, 5.4, 2.3]
        ]
        ```
        """
    )

    features_json = st.text_area(
        "JSON с признаками",
        value="[[5.1, 3.5, 1.4, 0.2]]",
        height=150,
    )

    if st.button("Получить предсказания"):
        try:
            features = json.loads(features_json)
            if not isinstance(features, list):
                raise ValueError("Корневой JSON должен быть списком.")
            for row in features:
                if not isinstance(row, list):
                    raise ValueError("Каждый объект должен быть списком чисел.")
        except Exception as exc:
            st.error(f"Ошибка парсинга JSON с признаками: {exc}")
            return

        payload = {"features": features}
        result = api_post_json(f"/models/{model_id}/predict", payload)
        if result is not None:
            preds = result.get("predictions", [])
            st.success(f"Предсказания: {preds}")



# Основной entrypoint Streamlit
def main() -> None:
    st.set_page_config(page_title="ML Homework Dashboard", layout="wide")
    st.title("ML Homework Dashboard")

    # Сайдбар: настройки backend URL
    st.sidebar.header("Настройки")
    current_url = get_backend_url()
    new_url = st.sidebar.text_input("Backend URL", current_url)
    if new_url != current_url:
        set_backend_url(new_url)

    st.sidebar.markdown(
        """
        **Подсказка:**
        По умолчанию ожидается, что REST-сервис запущен на
        `http://localhost:8000` командой:

        ```bash
        uvicorn ml_service.api_rest:app --host 0.0.0.0 --port 8000 --reload
        ```
        """
    )

    tabs = st.tabs(["Статус", "Датасеты", "Обучение", "Инференс"])

    with tabs[0]:
        render_status_tab()
    with tabs[1]:
        render_datasets_tab()
    with tabs[2]:
        render_training_tab()
    with tabs[3]:
        render_inference_tab()


if __name__ == "__main__":
    main()

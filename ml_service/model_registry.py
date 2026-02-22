# ml_service/model_registry.py
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class UnsupportedModelError(ValueError):
    """Исключение для неизвестных классов моделей."""


# Единый реестр доступных классов моделей.
# Здесь храним:
#  - описание (description)
#  - класс модели (cls)
#  - дефолтные гиперпараметры (default_hyperparams)
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "description": "Логистическая регрессия (sklearn.linear_model.LogisticRegression)",
        "cls": LogisticRegression,
        "default_hyperparams": {
            "C": 1.0,
            "max_iter": 200,
            "solver": "lbfgs",
            "random_state": 42,
        },
    },
    "random_forest": {
        "description": "Случайный лес (sklearn.ensemble.RandomForestClassifier)",
        "cls": RandomForestClassifier,
        "default_hyperparams": {
            "n_estimators": 100,
            "max_depth": None,
            "n_jobs": -1,
            "random_state": 42,
        },
    },
}


def get_available_model_classes() -> Dict[str, Dict[str, Any]]:
    """
    Вернуть словарь доступных классов моделей.

    Ключи = ID модели (используются в REST/gRPC/Streamlit),
    значения = словари с полями description, cls, default_hyperparams.
    """
    return MODEL_REGISTRY


def create_model_instance(model_class: str, hyperparams: Dict[str, Any] | None = None) -> Any:
    """
    Создать экземпляр модели по ID и гиперпараметрам.

    Если hyperparams не задан, берём дефолты из реестра.
    Если переданы hyperparams, они поверх дефолтов.
    """
    if model_class not in MODEL_REGISTRY:
        raise UnsupportedModelError(
            f"Неизвестный класс модели: {model_class!r}. "
            f"Доступные: {', '.join(MODEL_REGISTRY.keys())}"
        )

    meta = MODEL_REGISTRY[model_class]
    cls = meta["cls"]
    default_params = dict(meta.get("default_hyperparams") or {})
    user_params = dict(hyperparams or {})
    params = {**default_params, **user_params}
    return cls(**params)
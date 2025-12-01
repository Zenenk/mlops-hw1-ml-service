from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class UnsupportedModelError(ValueError):
    """Ошибка, если запрошен неизвестный класс модели."""


def get_available_model_classes() -> Dict[str, Dict[str, Any]]:
    """
    Возвращает словарь с описанием доступных классов моделей.

    Ключ — строковый идентификатор класса модели, который будет
    приходить снаружи (в API и gRPC).

    Значения содержат:
    - description: человекочитаемое описание модели;
    - default_hyperparams: словарь гиперпараметров по умолчанию.
    """
    return {
        "logistic_regression": {
            "description": (
                "Логистическая регрессия "
                "(sklearn.linear_model.LogisticRegression)"
            ),
            "default_hyperparams": {
                # Регуляризация
                "C": 1.0,
                # Максимальное число итераций
                "max_iter": 200,
                # Солвер по умолчанию
                "solver": "lbfgs",
                # Для воспроизводимости
                "random_state": 42,
            },
        },
        "random_forest": {
            "description": (
                "Случайный лес "
                "(sklearn.ensemble.RandomForestClassifier)"
            ),
            "default_hyperparams": {
                "n_estimators": 100,
                "max_depth": None,
                "n_jobs": -1,
                "random_state": 42,
            },
        },
    }


def create_model_instance(model_class: str, hyperparams: Dict[str, Any]):
    """
    Создаёт экземпляр ML-модели по идентификатору класса и гиперпараметрам.

    :param model_class: строковый идентификатор класса модели,
        один из ключей get_available_model_classes().
    :param hyperparams: словарь гиперпараметров. Здесь не валидируем
        жёстко состав словаря, если параметр неправильный,
        sklearn выбросит исключение.
    :return: не обученный экземпляр sklearn-модели.
    :raises UnsupportedModelError: если model_class неизвестен.
    """
    if model_class == "logistic_regression":
        return LogisticRegression(**hyperparams)

    if model_class == "random_forest":
        return RandomForestClassifier(**hyperparams)

    raise UnsupportedModelError(
        f"Неизвестный класс модели: {model_class!r}. "
        f"Доступные: {', '.join(get_available_model_classes().keys())}"
    )

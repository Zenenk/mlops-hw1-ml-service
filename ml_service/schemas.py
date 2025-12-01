from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

#Системные / служебные схемы
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    version: str = Field(..., example="0.1.0")


#Схемы для классов моделей
class ModelClassInfo(BaseModel):
    id: str = Field(..., example="logistic_regression")
    description: str = Field(
        ...,
        example="Логистическая регрессия (sklearn.linear_model.LogisticRegression)",
    )
    default_hyperparams: Dict[str, Any]


class ModelClassesResponse(BaseModel):
    classes: List[ModelClassInfo]


#Схемы для датасетов
class DatasetInfo(BaseModel):
    id: int = Field(..., example=1)
    name: str = Field(..., example="iris_v1.csv")
    path: str = Field(..., example="data/datasets/iris_v1.csv")
    description: Optional[str] = Field(
        None,
        example="Iris dataset (версии sklearn)",
    )
    version: Optional[str] = Field(
        None,
        example="dvc-tag-iris-v1",
    )


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]


#Схемы для обучения моделей
class TrainModelRequest(BaseModel):
    """
    Запрос на обучение новой модели.

    name: человекочитаемое имя модели
    model_class: идентификатор класса модели (logistic_regression / random_forest)
    dataset_id: ID датасета, на котором учим
    hyperparams: словарь гиперпараметров sklearn-модели
    """

    name: str = Field(..., example="my_logreg_model")
    model_class: str = Field(..., example="logistic_regression")
    dataset_id: int = Field(..., example=1)
    hyperparams: Dict[str, Any] = Field(
        default_factory=dict,
        example={"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
    )


class TrainModelResponse(BaseModel):
    model_id: int = Field(..., example=1)
    status: str = Field(..., example="trained")


#Схемы для описания модели и списка моделей
class ModelInfo(BaseModel):
    id: int = Field(..., example=1)
    name: str = Field(..., example="my_logreg_model")
    model_class: str = Field(..., example="logistic_regression")
    dataset_id: int = Field(..., example=1)
    status: str = Field(..., example="trained")
    hyperparams: Dict[str, Any] = Field(
        default_factory=dict,
        example={"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
    )
    clearml_model_id: Optional[str] = Field(
        None,
        example="4b2b2f1c0e7b4adca9b9a5a3a3c8f99a",
    )


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


#Схемы для инференса
class PredictRequest(BaseModel):
    """
    Запрос на предсказание.

    features: список объектов, каждый объект — список фичей (float).
    Размерность должна совпадать с обученным датасетом.
    """

    features: List[List[float]] = Field(
        ...,
        example=[[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]],
    )


class PredictResponse(BaseModel):
    """
    Ответ с предсказаниями модели.

    Для простоты — список классов (int). При желании можно расширить
    до вероятностей / top-k и т.д.
    """

    predictions: List[int] = Field(
        ...,
        example=[0, 2],
    )

class RetrainModelRequest(BaseModel):
    """
    Запрос на переобучение существующей модели.

    dataset_id в этом запросе не нужен: мы берём датасет из уже
    существующей записи модели в базе данных.
    """

    model_class: str = Field(..., example="random_forest")
    hyperparams: Dict[str, Any] = Field(
        default_factory=dict,
        example={"n_estimators": 200, "max_depth": 5},
    )


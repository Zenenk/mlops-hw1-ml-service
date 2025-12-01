from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Dataset(Base):
    """
    Таблица с датасетами.

    Храним:
    - id: числовой идентификатор
    - name: логическое имя / имя файла
    - path: полный путь к файлу на диске
    - description: произвольное описание
    - version: версия в DVC (по желанию, будем использовать позже)
    - created_at: дата и время создания записи
    """

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    path = Column(String, nullable=False)
    description = Column(String, nullable=True)
    version = Column(String, nullable=True)
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    models = relationship(
        "Model",
        back_populates="dataset",
        cascade="all,delete-orphan",
    )


class Model(Base):
    """
    Таблица с обученными моделями.

    Храним:
    - id: числовой идентификатор
    - name: человекочитаемое имя модели
    - model_class: класс модели (например, 'logistic_regression', 'random_forest')
    - dataset_id: внешний ключ на Dataset
    - status: статус ('training', 'trained', 'failed', 'deleted')
    - hyperparams: JSON со словарём гиперпараметров
    - clearml_model_id: идентификатор модели в ClearML (будем заполнять позже)
    - local_path: путь к локальному файлу с весами (joblib/pkl)
    - created_at: дата и время создания записи
    """

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    model_class = Column(String, nullable=False)

    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id"),
        nullable=False,
        index=True,
    )

    status = Column(
        String,
        default="trained",
        nullable=False,
    )

    hyperparams = Column(JSON, nullable=False)

    clearml_model_id = Column(String, nullable=True)

    local_path = Column(String, nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    dataset = relationship(
        "Dataset",
        back_populates="models",
    )

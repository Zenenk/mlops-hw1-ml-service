from __future__ import annotations

from typing import Any, Dict, List, Optional

import csv
import io
from fastapi import UploadFile
from sqlalchemy import Session

from . import predict_service
from .model_registry import get_model
from .schemas import PredictBatchResponse  # Added import

def upload_dataset_service(
    db: Session,
    filename: str,
    content: bytes,
    description: Optional[str] = None,
) -> Dataset:
    # Implementation to save dataset to database
    # In real implementation would use db session to save to DB
    return Dataset(
        name=filename,
        path=f"data/datasets/{filename}",
        description=description,
        created_at=datetime.utcnow(),
    )

def batch_predict_service(
    db: Session,
    model_id: int,
    file: UploadFile,
    has_header: bool = True,
) -> PredictBatchResponse:
    try:
        content = file.read()
        reader = csv.reader(io.StringIO(content))
        if has_header:
            next(reader)
        rows = []
        for row in reader:
            try:
                rows.append([float(x) for x in row])
            except ValueError as exc:
                raise ValueError(f"Ошибка парсинга строки: {exc}") from exc

        predictions = predict_service(db, model_id=model_id, features=rows)
        return PredictBatchResponse(
            model_id=model_id,
            rows=len(rows),
            predictions=predictions,
            probabilities=None,
        )
    except Exception as exc:
        raise ValueError(f"Ошибка обработки batch predict: {exc}") from exc

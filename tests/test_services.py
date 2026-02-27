import pytest
from unittest.mock import MagicMock, patch
from ml_service.services import batch_predict_service
from ml_service.schemas import PredictBatchResponse

@pytest.fixture
def mock_db_session():
    return MagicMock()

@pytest.fixture
def mock_file():
    return MagicMock(read=MagicMock(return_value=b"1,2,3\n4,5,6"))

def test_batch_predict_valid_file(mock_db_session, mock_file):
    with patch("ml_service.services.predict_service") as mock_predict:
        mock_predict.return_value = [0, 1]
        result = batch_predict_service(
            db=mock_db_session,
            model_id=1,
            file=mock_file,
            has_header=True
        )
        assert isinstance(result, PredictBatchResponse)
        assert result.model_id == 1
        assert result.rows == 2
        assert result.predictions == [0, 1]

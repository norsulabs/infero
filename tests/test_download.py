import os
import pytest
from infero.pull.download import check_model_integrity


@pytest.fixture
def setup_model_directory():
    model_name = "test-model"
    model_dir = f"infero/data/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    yield model_name
    # Cleanup after test
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, file))
        os.rmdir(model_dir)


def test_check_model_integrity(setup_model_directory):
    model_name = setup_model_directory
    assert check_model_integrity(model_name) is True

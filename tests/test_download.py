import pytest
import os
from unittest.mock import patch
from infero.pull.download import download_file


@patch("infero.pull.download.download_file")
def test_download_file_success(mock_download_file):
    mock_download_file.return_value = True
    url = "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment/blob/main/vocab.json"
    result = download_file(url, "file.json")
    if os.path.exists("file.json"):
        os.remove("file.json")
    assert result


if __name__ == "__main__":
    pytest.main()

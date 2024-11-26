import os
import requests
import json
import yaml
from tqdm import tqdm

from infero.utils import print_error, print_success, sanitize_model_name, print_neutral


def is_supported(model: str):
    config_url = f"https://huggingface.co/{model}/raw/main/config.json"
    config_file = requests.get(config_url)
    config = json.loads(config_file.content)
    arch = config.get("architectures", [None])[0]
    if not arch:
        print_error(f"Architecture not specified in config for {model}")
        return False
    script_dir = os.path.dirname(__file__)
    supported_architectures_path = os.path.join(
        script_dir, "supported_architectures.yaml"
    )

    with open(supported_architectures_path, "r") as file:
        archs = yaml.safe_load(file)
    return any(arch == item["name"] for item in archs["architectures"])


def check_model_integrity(model: str):
    model_path = f"infero/data/models/{sanitize_model_name(model)}/pytorch_model.bin"
    vocab_path = f"infero/data/models/{sanitize_model_name(model)}/vocab.json"
    vocab_path_2 = f"infero/data/models/{sanitize_model_name(model)}/vocab.txt"
    config_path = f"infero/data/models/{sanitize_model_name(model)}/config.json"

    if not os.path.exists(model_path):
        print_neutral(f"Model {model} not found, downloading...")
        download_model(model)

    if not os.path.exists(vocab_path) and not os.path.exists(vocab_path_2):
        print_neutral(f"Vocab file for {model} not found, downloading...")
        vocab_url = f"https://huggingface.co/{model}/resolve/main/vocab.json"
        if not download_file(vocab_url, vocab_path):
            vocab_url = f"https://huggingface.co/{model}/resolve/main/vocab.txt"
            download_file(vocab_url, vocab_path_2)

    if not os.path.exists(config_path):
        print_neutral(f"Config file for {model} not found, downloading...")
        config_url = f"https://huggingface.co/{model}/raw/main/config.json"
        download_file(config_url, config_path)

    return True


def download_file(url, output_path):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 8192
            t = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    t.update(len(chunk))
                    f.write(chunk)
            t.close()
            if total_size != 0 and t.n != total_size:
                print_error(f"Failed to download file from {url}")
                os.remove(output_path)
                return False
        return True
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to download file from {url}: {e}")
        return False


def download_model(model: str):
    model_url = f"https://huggingface.co/{model}/resolve/main/pytorch_model.bin"
    vocab_url = f"https://huggingface.co/{model}/resolve/main/vocab.json"
    config_url = f"https://huggingface.co/{model}/raw/main/config.json"
    output_dir = f"infero/data/models/{sanitize_model_name(model)}"
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    vocab_path = os.path.join(output_dir, "vocab.json")
    config_path = os.path.join(output_dir, "config.json")

    os.makedirs(output_dir, exist_ok=True)

    if download_file(model_url, model_path):
        print_success(f"Model {model} downloaded successfully")
    else:
        return

    if download_file(vocab_url, vocab_path):
        print_success(f"Vocab file for {model} downloaded successfully")
    else:
        vocab_url = f"https://huggingface.co/{model}/resolve/main/vocab.txt"
        vocab_path = os.path.join(output_dir, "vocab.txt")
        if download_file(vocab_url, vocab_path):
            print_success(f"Vocab file for {model} downloaded successfully")
        else:
            print_error(f"Failed to download vocab file for {model}")

    if download_file(config_url, config_path):
        print_success(f"Config file for {model} downloaded successfully")
    else:
        print_error(f"Failed to download config file for {model}")


def check_model(model: str):
    if is_supported(model):
        print_success(f"Model {model} is supported")
    else:
        print_error("Model architecture not supported")

    if os.path.exists(f"infero/data/models/{sanitize_model_name(model)}"):
        print_success(f"Model {model} already exists")
        chk = check_model_integrity(model)
        if chk is True:
            return True
    else:
        download_model(model)
        return True

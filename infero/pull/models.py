import os
from infero.utils import (
    sanitize_model_name,
    print_success,
    print_error,
    get_models_dir,
    print_neutral,
    unsanitize_model_name,
)
from tabulate import tabulate
import shutil


def remove_model(model):
    model_path = os.path.join(get_models_dir(), sanitize_model_name(model))
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print_success(f"Model {model} removed")
    else:
        print_error(f"Model {model} not found")


def display_models():

    if not os.path.exists(get_models_dir()):
        print_neutral("No models found")
        return
    models_dir = get_models_dir()
    models = []
    for model in os.listdir(models_dir):
        quantized = (
            f"{os.path.getsize(os.path.join(models_dir, model, 'model_quantized.onnx')) / 1024 / 1024:.2f}"
            if os.path.exists(os.path.join(models_dir, model, "model_quantized.onnx"))
            else ""
        )
        size = (
            os.path.getsize(os.path.join(models_dir, model, "pytorch_model.bin"))
            / 1024
            / 1024
        )
        models.append([unsanitize_model_name(model), size, quantized])
    table = tabulate(
        models, headers=["Name", "Size (MB)", "Quantized (MB)"], tablefmt="grid"
    )
    print_neutral(table)

import os
from infero.utils import sanitize_model_name, print_success, print_error


def remove_model(model):
    model_path = os.path.join(
        os.getcwd(), "infero/data/models", sanitize_model_name(model)
    )
    if os.path.exists(model_path):
        os.rmdir(model_path)
        print_success(f"Model {model} removed")
    else:
        print_error(f"Model {model} not found")

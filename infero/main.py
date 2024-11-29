import os
import subprocess
import typer
from infero.pull.download import check_model, pull_model
from infero.convert.onnx import convert_to_onnx, convert_to_onnx_q8
from infero.utils import (
    sanitize_model_name,
    get_models_dir,
    get_package_dir,
    print_success_bold,
    print_error,
)
from infero.pull.models import remove_model, display_models

app = typer.Typer(name="infero")


@app.command("run")
def run(model: str, quantize: bool = False):
    convert_to_onnx(model)
    if quantize:
        convert_to_onnx_q8(model)
    if check_model(model):
        model_path = os.path.join(get_models_dir(), sanitize_model_name(model))
        package_dir = get_package_dir()
        server_script_path = os.path.join(package_dir, "serve", "server.py")
        subprocess.run(
            ["python", server_script_path, model_path, str(quantize).lower()]
        )
    else:
        typer.echo("Failed to run model")


@app.command("pull")
def pull(model: str, quantize: bool = False):
    if pull_model(model):
        print_success_bold(f"Model {model} pulled successfully")
    else:
        print_error("Failed to get model")


@app.command("list")
def list_models():
    display_models()


@app.command("remove")
def remove(model: str):
    remove_model(model)


if __name__ == "__main__":
    app()

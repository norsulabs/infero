import os
import subprocess
import typer
from infero.pull.download import check_model
from infero.convert.onnx import convert_to_onnx
from infero.utils import sanitize_model_name
from infero.pull.models import remove_model

app = typer.Typer(name="infero")


@app.command("run")
def pull(model: str):
    if check_model(model):
        convert_to_onnx(model)
        model_path = f"infero/data/models/{sanitize_model_name(model)}"
        script_dir = os.path.dirname(__file__)
        server_script_path = os.path.join(script_dir, "serve", "server", "server.py")
        subprocess.run(["python", server_script_path, model_path])
    else:
        typer.echo("Failed to run model")


@app.command("list")
def list_models():
    models = os.path.join(os.getcwd(), "infero/data/models")
    for model in os.listdir(models):
        typer.echo(model)


@app.command("remove")
def remove(model: str):
    remove_model(model)


if __name__ == "__main__":
    app()

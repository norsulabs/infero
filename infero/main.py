import subprocess
import typer
from infero.pull.download import check_model
from infero.convert.onnx import convert_to_onnx
from infero.utils import sanitize_model_name

app = typer.Typer(name="infero")


@app.command("run")
def pull(model: str):
    if check_model(model):
        convert_to_onnx(model)
        model_path = f"infero/data/models/{sanitize_model_name(model)}"
        subprocess.run(["python", "infero/serve/server.py", model_path])
    else:
        typer.echo("Failed to run model")


@app.command("list")
def list_models():
    typer.echo("List of models")


if __name__ == "__main__":
    app()

import typer
from infero.pull.download import check_model
from infero.convert.onnx import convert_to_onnx

app = typer.Typer(name="Infero")


@app.command("pull")
def pull(model: str, quantization: bool = False):
    if check_model(model):
        convert_to_onnx(model)
    else:
        typer.echo("Failed to run model")
    # if quantization:
    #     quantize_model(model)


@app.command("list")
def list_models():
    typer.echo("List of models")


if __name__ == "__main__":
    app()

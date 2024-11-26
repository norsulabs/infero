import typer


def print_success(message: str):
    typer.echo(typer.style(message, fg=typer.colors.GREEN))


def print_error(message: str):
    typer.echo(typer.style(message, fg=typer.colors.RED))


def print_success_bold(message: str):
    typer.echo(typer.style(message, fg=typer.colors.GREEN, bold=True))


def print_neutral(message: str):
    typer.echo(typer.style(message, fg=typer.colors.BLUE))


def sanitize_model_name(model: str):
    return model.replace("/", "_")


def unsanitize_model_name(model: str):
    return model.replace("_", "/")

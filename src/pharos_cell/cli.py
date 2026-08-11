"""Top-level PHAROS command-line interface."""

from typing import Annotated

import typer

from pharos_cell import __version__

app = typer.Typer(
    name="pharos",
    help="Target-directed drug-combination screening from single-cell state embeddings.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pharos {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the PHAROS version and exit.",
        ),
    ] = False,
) -> None:
    """Run PHAROS workflows."""


@app.command(
    "open-search",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def open_search(ctx: typer.Context) -> None:
    """Search for drug combinations that convert a starting cell state."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.open_search import main as open_search_main

    open_search_main(argv)


def main() -> None:
    """Run the installed ``pharos`` console command."""

    app(prog_name="pharos")

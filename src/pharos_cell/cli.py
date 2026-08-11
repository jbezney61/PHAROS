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

admissibility_app = typer.Typer(
    help="Check whether a proposed cell-state conversion is suitable for PHAROS.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
app.add_typer(admissibility_app, name="admissibility")


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


@admissibility_app.command(
    "calibrate",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def admissibility_calibrate(ctx: typer.Context) -> None:
    """Check STATE predictions against observed perturbation targets."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.admissibility.calibration_cli import main as calibration_main

    calibration_main(argv)


@admissibility_app.command(
    "manifold",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def admissibility_manifold(ctx: typer.Context) -> None:
    """Build or query the reference embedding manifold."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.admissibility.manifold_cli import main as manifold_main

    manifold_main(argv)


@admissibility_app.command(
    "separation",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def admissibility_separation(ctx: typer.Context) -> None:
    """Screen start/target state separation before a PHAROS search."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.admissibility.separation_cli import main as separation_main

    separation_main(argv)


def main() -> None:
    """Run the installed ``pharos`` console command."""

    app(prog_name="pharos")

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

hypothesis_app = typer.Typer(
    help="Evaluate specified drug combinations against random controls.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
app.add_typer(hypothesis_app, name="hypothesis-driven")

report_app = typer.Typer(
    help="Generate reports from completed PHAROS runs.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
app.add_typer(report_app, name="report")

evaluate_app = typer.Typer(
    help="Evaluate recovery and statistical significance in PHAROS results.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
app.add_typer(evaluate_app, name="evaluate")

models_app = typer.Typer(
    help="Download and verify pretrained models used by PHAROS.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
app.add_typer(models_app, name="models")


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


@hypothesis_app.command(
    "pair",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def hypothesis_pair(ctx: typer.Context) -> None:
    """Evaluate one named, fixed-drug, or MOA-defined candidate pair."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.hypothesis.pair_cli import main as pair_main

    pair_main(argv)


@hypothesis_app.command(
    "panel",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def hypothesis_panel(ctx: typer.Context) -> None:
    """Evaluate a file containing a panel of explicit drug pairs."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.hypothesis.panel_cli import main as panel_main

    panel_main(argv)


@hypothesis_app.command(
    "summarize",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def hypothesis_summarize(ctx: typer.Context) -> None:
    """Collate and statistically compare multiple pair-analysis runs."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.hypothesis.reports.multi import main as summarize_main

    summarize_main(argv)


@report_app.command(
    "open-search",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def report_open_search(ctx: typer.Context) -> None:
    """Audit where a known drug pair appears in open-search results."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.reports.open_search import main as report_main

    report_main(argv)


@evaluate_app.command(
    "pair-recovery",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def evaluate_pair_recovery(ctx: typer.Context) -> None:
    """Test target-pair recovery against a beam-matched Monte Carlo null."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.evaluation.pair_recovery_cli import main as pair_recovery_main

    exit_code = pair_recovery_main(argv)
    if exit_code:
        raise typer.Exit(exit_code)


@models_app.command(
    "download",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def models_download(ctx: typer.Context) -> None:
    """Download pinned SE-600M and ST-SE-Tahoe model artifacts."""

    argv = list(ctx.args) or ["--help"]
    from pharos_cell.model_download import main as download_main

    download_main(argv)


def main() -> None:
    """Run the installed ``pharos`` console command."""

    app(prog_name="pharos")

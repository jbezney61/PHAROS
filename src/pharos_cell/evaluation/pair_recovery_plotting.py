"""Plotting helpers for per-run target-pair recovery tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


PUBLICATION_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 360,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

HEADER_FACE = "#2f3e46"
HEADER_TEXT = "#ffffff"
ROW_FACE_A = "#ffffff"
ROW_FACE_B = "#f4f7f8"
GRID = "#d7dee3"
TITLE_COLOR = "#1b262c"


def require_plotting():
    import matplotlib.pyplot as plt

    plt.rcParams.update(PUBLICATION_RC)
    return plt


def savefig_bundle(fig: Any, png_path: Path) -> Tuple[Path, Path]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def plot_per_run_results_table(
    display: pd.DataFrame,
    output_dir: Path,
    title: str = "Target-pair recovery (Random model MC null)",
) -> Tuple[Path, Path]:
    plt = require_plotting()
    n_rows = max(len(display), 1)
    fig_h = max(2.1, 0.78 * n_rows + 1.15)
    fig_w = 11.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, pad=4, color=TITLE_COLOR, fontweight="semibold", fontsize=12)

    col_labels = [
        "Target pair",
        "# appearances of A",
        "# appearances of B",
        "Exact pair found",
        "Exact pair OT rank",
    ]
    cell_text = []
    for _, row in display.iterrows():
        cell_text.append(
            [
                str(row["target_pair"]),
                f"{row['n_A']}\n{row['n_A_p']}",
                f"{row['n_B']}\n{row['n_B_p']}",
                f"{row['exact']}\n{row['exact_p']}",
                str(row["best_exact_rank"]),
            ]
        )

    table = ax.table(
        cellText=cell_text if cell_text else [["—"] * len(col_labels)],
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        bbox=[0.0, 0.08, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)

    # Wider first column for full drug names; wrap-friendly header heights.
    col_widths = [0.30, 0.175, 0.175, 0.185, 0.165]
    for (row_i, col_i), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.set_width(col_widths[col_i])
        if row_i == 0:
            cell.set_facecolor(HEADER_FACE)
            cell.set_text_props(color=HEADER_TEXT, weight="bold", fontsize=8.5)
            cell.set_height(0.22)
        else:
            cell.set_facecolor(ROW_FACE_A if row_i % 2 else ROW_FACE_B)
            cell.set_height(0.26)
            if col_i == 0:
                cell.get_text().set_ha("left")
                cell.PAD = 0.03

    # Footnote clarifying A/B
    ax.text(
        0.0,
        0.01,
        "A and B are the first and second drugs of the target pair, respectively.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4a5560",
        style="italic",
    )

    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.02, right=0.98)
    return savefig_bundle(fig, Path(output_dir) / "figures" / "per_run_results_table.png")


def make_report_plots(analysis: Dict[str, Any]) -> Dict[str, Tuple[Path, Path]]:
    return {
        "per_run_results_table": plot_per_run_results_table(
            display=analysis["per_run_results_display"],
            output_dir=analysis["output_dir"],
        )
    }

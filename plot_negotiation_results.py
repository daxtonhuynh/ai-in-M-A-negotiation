from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from PIL import Image, ImageOps, ImageDraw

MAIN_METHOD_ORDER = [
    "minimax",
    "alpha_beta",
    "hab_h1_d2",
    "hab_h2_d2",
    "hab_h3_d2",
    "hab_h3_d3",
]

ALL_METHOD_ORDER = [
    "minimax",
    "alpha_beta",
    "alpha_beta_unordered",
    "hab_h1_d2",
    "hab_h2_d2",
    "hab_h3_d2",
    "hab_h3_d3",
    "hab_h3_d3_unordered",
]

DISPLAY_LABELS = {
    "minimax": "Minimax",
    "alpha_beta": "Alpha-Beta\n(ordered)",
    "alpha_beta_unordered": "Alpha-Beta\n(unordered)",
    "hab_h1_d2": "HAB h1\n(d=2)",
    "hab_h2_d2": "HAB h2\n(d=2)",
    "hab_h3_d2": "HAB h3\n(d=2)",
    "hab_h3_d3": "HAB h3\n(d=3)",
    "hab_h3_d3_unordered": "HAB h3\n(d=3, unord.)",
}

CLASS_ORDER = ["buyer_advantaged", "balanced", "seller_advantaged"]
CLASS_LABELS = {
    "buyer_advantaged": "Buyer-advantaged",
    "balanced": "Balanced",
    "seller_advantaged": "Seller-advantaged",
}

MOVE_ORDERING_FAMILY_ORDER = ["alpha_beta", "hab_h3_d3"]
MOVE_ORDERING_FAMILY_LABELS = {
    "alpha_beta": "Alpha-Beta",
    "hab_h3_d3": "HAB h3 (d=3)",
}

# final muted palette
BASE_BAR = "#86A9D7"       # muted blue
POS_BAR = "#BFDDBE"        # light green
NEG_BAR = "#EBC1BC"        # light red
ORDERED_BAR = "#7FA8D1"    # muted blue
UNORDERED_BAR = "#D8BE96"  # soft sand / desaturated orange
HEATMAP_CMAP = "YlGnBu"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def label_methods(df: pd.DataFrame, column: str = "method_label") -> pd.DataFrame:
    df = df.copy()
    labels = df[column].astype(str)
    df["plot_label"] = labels.map(DISPLAY_LABELS).fillna(labels)
    return df


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_bar_labels(ax: plt.Axes, fmt: str = "{:.2f}", percent: bool = False, pad_frac: float = 0.015) -> None:
    ylim = ax.get_ylim()
    y_offset = (ylim[1] - ylim[0]) * pad_frac
    for patch in ax.patches:
        height = patch.get_height()
        if math.isnan(height):
            continue
        text = f"{height:.0%}" if percent else fmt.format(height)
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + y_offset,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_simple_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    path: Path,
    method_order: Iterable[str],
    percent: bool = False,
    zero_center: bool = False,
    color_by_sign: bool = False,
    figsize: tuple[float, float] = (11.0, 5.6),
    base_color: str = BASE_BAR,
) -> None:
    plot_df = df[df["method_label"].isin(method_order)].copy()
    plot_df["method_label"] = pd.Categorical(plot_df["method_label"], categories=list(method_order), ordered=True)
    plot_df = plot_df.sort_values("method_label")
    plot_df = label_methods(plot_df)

    values = plot_df[value_col].astype(float).to_numpy()
    if color_by_sign:
        colors_list = [POS_BAR if value >= 0 else NEG_BAR for value in values]
    else:
        colors_list = [base_color] * len(values)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(plot_df["plot_label"], values, color=colors_list, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    style_axis(ax)

    if percent:
        upper = max(1.08, float(np.nanmax(values)) * 1.12)
        ax.set_ylim(0, upper)
        add_bar_labels(ax, percent=True)
    else:
        if zero_center:
            bound = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))))
            ax.set_ylim(-bound * 1.22, bound * 1.22)
            ax.axhline(0, color="#666666", linewidth=1.0)
        add_bar_labels(ax, fmt="{:.2f}")

    fig.tight_layout(pad=1.2)
    savefig(fig, path)


def _text_color_for_rgba(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.57 else "white"


def plot_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    cbar_label: str,
    path: Path,
    fmt: str = "{:.2f}",
    cmap: str = HEATMAP_CMAP,
) -> None:
    matrix = pivot_df.to_numpy(dtype=float)
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_obj, norm=norm)

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels([DISPLAY_LABELS.get(col, col) for col in pivot_df.columns], fontsize=10)
    ax.set_yticklabels([CLASS_LABELS.get(idx, idx) for idx in pivot_df.index], fontsize=10)

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "NA" if np.isnan(value) else fmt.format(value)
            rgba = cmap_obj(norm(value)) if not np.isnan(value) else (1, 1, 1, 1)
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=_text_color_for_rgba(rgba))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)
    fig.tight_layout(pad=1.2)
    savefig(fig, path)


def plot_move_ordering_grouped(
    move_df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    path: Path,
    percent: bool = False,
) -> None:
    plot_df = move_df[move_df["method_family"].isin(MOVE_ORDERING_FAMILY_ORDER)].copy()
    plot_df["method_family"] = pd.Categorical(
        plot_df["method_family"], categories=MOVE_ORDERING_FAMILY_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values(["method_family", "ordering_variant"])

    families = [fam for fam in MOVE_ORDERING_FAMILY_ORDER if fam in plot_df["method_family"].astype(str).tolist()]
    ordered_vals = []
    unordered_vals = []
    for family in families:
        sub = plot_df[plot_df["method_family"] == family]
        ordered_vals.append(float(sub.loc[sub["ordering_variant"] == "ordered", value_col].iloc[0]))
        unordered_vals.append(float(sub.loc[sub["ordering_variant"] == "unordered", value_col].iloc[0]))

    x = np.arange(len(families))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    bars1 = ax.bar(x - width / 2, ordered_vals, width, label="Ordered", color=ORDERED_BAR, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, unordered_vals, width, label="Unordered", color=UNORDERED_BAR, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([MOVE_ORDERING_FAMILY_LABELS.get(fam, fam) for fam in families], fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)

    if percent:
        ymax = max(max(ordered_vals), max(unordered_vals))
        ax.set_ylim(0, ymax * 1.2)
        for bars in (bars1, bars2):
            for patch in bars:
                height = patch.get_height()
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    height + ymax * 0.025,
                    f"{height:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    else:
        ymax = max(max(ordered_vals), max(unordered_vals))
        ax.set_ylim(0, ymax * 1.2)
        for bars in (bars1, bars2):
            for patch in bars:
                height = patch.get_height()
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    height + ymax * 0.025,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    fig.tight_layout(pad=1.2)
    savefig(fig, path)


def build_contact_sheet(image_paths: List[Path], output_path: Path, thumb_width: int = 510) -> None:
    if not image_paths:
        return

    thumbs = []
    header_h = 28
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width, 1000))
        framed = ImageOps.expand(image, border=8, fill="white")
        canvas = Image.new("RGB", (framed.width, framed.height + header_h), color=(245, 245, 245))
        canvas.paste(framed, (0, header_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), path.stem, fill="black")
        thumbs.append(canvas)

    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    max_w = max(img.width for img in thumbs)
    max_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (cols * max_w + 20, rows * max_h + 20), color=(245, 245, 245))

    for idx, image in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = 10 + col * max_w
        y = 10 + row * max_h
        sheet.paste(image, (x, y))

    sheet.save(output_path)


def main() -> None:
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    input_dir = base_dir / "outputs"
    output_dir = input_dir / "plots"
    ensure_dir(output_dir)

    summary_method = pd.read_csv(input_dir / "negotiation_batch_summary_by_method.csv")
    summary_class_method = pd.read_csv(input_dir / "negotiation_batch_summary_by_class_method.csv")
    summary_move_ordering = pd.read_csv(input_dir / "negotiation_batch_summary_move_ordering.csv")

    summary_method = summary_method.copy()
    summary_method["mean_runtime_ms"] = summary_method["mean_runtime_sec"] * 1000.0

    benchmark_df = summary_method[summary_method["method_label"].isin(MAIN_METHOD_ORDER)].copy()
    benchmark_df["method_label"] = pd.Categorical(
        benchmark_df["method_label"], categories=MAIN_METHOD_ORDER, ordered=True
    )
    benchmark_df = benchmark_df.sort_values("method_label")
    benchmark_table = pd.DataFrame(
        {
            "Method": [DISPLAY_LABELS.get(label, label).replace("\n", " ") for label in benchmark_df["method_label"]],
            "Runtime (ms)": (benchmark_df["mean_runtime_ms"]).round(3),
            "Nodes": benchmark_df["mean_nodes"].round(2),
            "Cutoff rate": benchmark_df["mean_cutoff_rate"].round(3),
            "Deal success": benchmark_df["mean_deal_success"].round(3),
            "Mean utility": benchmark_df["mean_utility"].round(3),
            "Mean rounds": benchmark_df["mean_rounds"].round(3),
        }
    )
    benchmark_table.to_csv(output_dir / "table_01_overall_benchmark.csv", index=False)

    fig, ax = plt.subplots(figsize=(12.5, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=benchmark_table.values,
        colLabels=benchmark_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    ax.set_title("Overall Benchmark Summary", fontsize=15, pad=8)
    fig.tight_layout(pad=1.0)
    savefig(fig, output_dir / "table_01_overall_benchmark.png")

    # separate overview figures 01-04
    plot_simple_bar(
        benchmark_df,
        "mean_runtime_ms",
        "Overall Runtime by Method",
        "Runtime (ms)",
        output_dir / "figure_01_runtime_by_method.png",
        MAIN_METHOD_ORDER,
    )
    plot_simple_bar(
        benchmark_df,
        "mean_nodes",
        "Overall Nodes Expanded by Method",
        "Mean nodes expanded",
        output_dir / "figure_02_nodes_by_method.png",
        MAIN_METHOD_ORDER,
    )
    plot_simple_bar(
        benchmark_df,
        "mean_cutoff_rate",
        "Overall Cutoff Rate by Method",
        "Cutoff-event rate",
        output_dir / "figure_03_cutoff_rate_by_method.png",
        MAIN_METHOD_ORDER,
        percent=True,
    )
    plot_simple_bar(
        benchmark_df,
        "mean_utility",
        "Overall Utility by Method",
        "Mean utility (buyer/MAX perspective)",
        output_dir / "figure_04_utility_by_method.png",
        MAIN_METHOD_ORDER,
        zero_center=True,
    )

    class_method_main = summary_class_method[summary_class_method["method_label"].isin(MAIN_METHOD_ORDER)].copy()
    class_method_main["scenario_class"] = pd.Categorical(
        class_method_main["scenario_class"], categories=CLASS_ORDER, ordered=True
    )
    class_method_main["method_label"] = pd.Categorical(
        class_method_main["method_label"], categories=MAIN_METHOD_ORDER, ordered=True
    )
    class_method_main = class_method_main.sort_values(["scenario_class", "method_label"])

    utility_heatmap = class_method_main.pivot(
        index="scenario_class",
        columns="method_label",
        values="mean_utility",
    ).reindex(index=CLASS_ORDER, columns=MAIN_METHOD_ORDER)
    plot_heatmap(
        utility_heatmap,
        "Utility by Scenario Class and Method",
        "Mean utility",
        output_dir / "figure_05_utility_heatmap_by_class_method.png",
        fmt="{:.2f}",
        cmap=HEATMAP_CMAP,
    )

    rounds_heatmap = class_method_main.pivot(
        index="scenario_class",
        columns="method_label",
        values="mean_rounds",
    ).reindex(index=CLASS_ORDER, columns=MAIN_METHOD_ORDER)
    plot_heatmap(
        rounds_heatmap,
        "Mean Negotiation Rounds by Scenario Class and Method",
        "Mean rounds",
        output_dir / "figure_06_rounds_heatmap_by_class_method.png",
        fmt="{:.2f}",
        cmap=HEATMAP_CMAP,
    )

    relative_df = summary_method[summary_method["method_label"].isin(ALL_METHOD_ORDER)].copy()
    relative_df["method_label"] = pd.Categorical(
        relative_df["method_label"], categories=ALL_METHOD_ORDER, ordered=True
    )
    relative_df = relative_df.sort_values("method_label")
    non_minimax_order = [label for label in ALL_METHOD_ORDER if label != "minimax"]

    plot_simple_bar(
        relative_df,
        "mean_match_minimax_root_action",
        "Root Action Match vs Minimax",
        "Match rate",
        output_dir / "figure_07_root_action_match_vs_minimax.png",
        non_minimax_order,
        percent=True,
    )

    plot_simple_bar(
        relative_df,
        "mean_utility_gap_vs_minimax",
        "Utility Gap vs Minimax",
        "Utility gap relative to minimax",
        output_dir / "figure_08_utility_gap_vs_minimax.png",
        non_minimax_order,
        zero_center=True,
        color_by_sign=True,
    )

    plot_simple_bar(
        relative_df,
        "mean_price_gap_vs_minimax",
        "Final Price Gap vs Minimax",
        "Price gap relative to minimax",
        output_dir / "figure_09_price_gap_vs_minimax.png",
        non_minimax_order,
        zero_center=True,
        color_by_sign=True,
    )

    summary_move_ordering = summary_move_ordering.copy()
    summary_move_ordering["mean_runtime_ms"] = summary_move_ordering["mean_runtime_sec"] * 1000.0

    plot_move_ordering_grouped(
        summary_move_ordering,
        "mean_nodes",
        "Nodes Expanded: Ordered vs Unordered",
        "Mean nodes expanded",
        output_dir / "figure_10_move_ordering_nodes.png",
    )

    plot_move_ordering_grouped(
        summary_move_ordering,
        "mean_runtime_ms",
        "Runtime: Ordered vs Unordered",
        "Runtime (ms)",
        output_dir / "figure_11_move_ordering_runtime.png",
    )

    plot_move_ordering_grouped(
        summary_move_ordering,
        "mean_match_minimax_root_action",
        "Root Action Match: Ordered vs Unordered",
        "Match rate vs minimax",
        output_dir / "figure_12_move_ordering_root_match.png",
        percent=True,
    )

    image_paths = [
        output_dir / "table_01_overall_benchmark.png",
        output_dir / "figure_01_runtime_by_method.png",
        output_dir / "figure_02_nodes_by_method.png",
        output_dir / "figure_03_cutoff_rate_by_method.png",
        output_dir / "figure_04_utility_by_method.png",
        output_dir / "figure_05_utility_heatmap_by_class_method.png",
        output_dir / "figure_06_rounds_heatmap_by_class_method.png",
        output_dir / "figure_07_root_action_match_vs_minimax.png",
        output_dir / "figure_08_utility_gap_vs_minimax.png",
        output_dir / "figure_09_price_gap_vs_minimax.png",
        output_dir / "figure_10_move_ordering_nodes.png",
        output_dir / "figure_11_move_ordering_runtime.png",
        output_dir / "figure_12_move_ordering_root_match.png",
    ]
    build_contact_sheet([p for p in image_paths if p.exists()], output_dir / "contact_sheet.png")

    bundle_path = base_dir / "5470_plots_bundle.zip"
    if bundle_path.exists():
        bundle_path.unlink()
    shutil.make_archive(str(bundle_path.with_suffix("")), "zip", output_dir)

    print("All plots created successfully.")
    print(f"Output folder: {output_dir}")
    print(f"Bundle: {bundle_path}")


if __name__ == "__main__":
    main()

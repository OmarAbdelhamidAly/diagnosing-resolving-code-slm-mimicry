"""Publication-quality visualization functions for Reduction Ladder evaluations."""

import os
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd


def plot_single_model_degradation(
    level_reports: Dict[str, Any],
    model_name: str,
    output_filepath: str = "results/baseline/degradation_curve.png"
) -> str:
    """Plot Pass@1 and Pass@5 curves for a single model across L0-L5."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    level_labels = ["L0\n(HumanEval)", "L1\n(Subtle)", "L2\n(ToolUse)", "L3\n(Creative)", "L4\n(Difficult)", "L5\n(Combine)"]

    p1_scores = [level_reports.get(lvl, {}).get("pass_at_1", 0.0) * 100 for lvl in levels if lvl in level_reports]
    
    plt.figure(figsize=(9, 5), dpi=300)
    plt.axhline(50.0, color="red", linestyle="--", alpha=0.7, label="Collapse Threshold (50%)")
    plt.plot(levels[:len(p1_scores)], p1_scores, marker="o", color="#FF6400", linewidth=2.5, label="Pass@1 (Greedy)")

    # If Pass@5 is available
    if any(level_reports.get(lvl, {}).get("pass_at_5") is not None for lvl in levels):
        p5_scores = [level_reports.get(lvl, {}).get("pass_at_5", 0.0) * 100 for lvl in levels if lvl in level_reports and level_reports[lvl].get("pass_at_5") is not None]
        if len(p5_scores) == len(p1_scores):
            plt.plot(levels[:len(p5_scores)], p5_scores, marker="s", color="#1E78C8", linestyle="--", linewidth=2.0, label="Pass@5 (Sampling T=0.8)")

    plt.title(f"{model_name}: Reduction Ladder Degradation Curve", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Ladder Level", fontsize=10, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    plt.xticks(range(len(p1_scores)), level_labels[:len(p1_scores)])
    plt.ylim(-2, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    return output_filepath


def plot_error_taxonomy(
    level_reports: Dict[str, Any],
    model_name: str,
    output_filepath: str = "results/baseline/error_taxonomy.png"
) -> str:
    """Plot stacked bar chart of error taxonomy categories across levels."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    records = []
    for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        if lvl in level_reports:
            breakdown = level_reports[lvl].get("error_breakdown", {}).copy()
            breakdown["Level"] = lvl
            records.append(breakdown)

    if not records:
        return output_filepath

    df = pd.DataFrame(records).set_index("Level").fillna(0)
    # Drop "pass" from error breakdown
    err_df = df.drop(columns=["pass"], errors="ignore")

    plt.figure(figsize=(9, 5), dpi=300)
    err_df.plot(
        kind="bar",
        stacked=True,
        colormap="Set2",
        figsize=(9, 5),
        edgecolor="black",
        linewidth=0.5
    )
    plt.title(f"{model_name}: Diagnostic Error Taxonomy Breakdown", fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Ladder Level", fontsize=10, fontweight="bold")
    plt.ylabel("Number of Tasks", fontsize=10, fontweight="bold")
    plt.xticks(rotation=0)
    plt.legend(title="Error Category", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    return output_filepath


def plot_multi_model_comparison(
    models_data: Dict[str, Dict[str, Any]],
    output_filepath: str = "results/multi_model_comparison.png"
) -> str:
    """Plot comparative degradation curves for multiple models (M1 through M6)."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    level_labels = ["L0 (HumanEval)", "L1 (Subtle)", "L2 (ToolUse)", "L3 (Creative)", "L4 (Difficult)", "L5 (Combine)"]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.axhline(50.0, color="red", linestyle="--", alpha=0.7, label="Collapse Threshold (50%)")

    colors = ["#FF6400", "#1E78C8", "#9B51E0", "#6C757D", "#008080", "#28A745"]
    markers = ["o", "s", "d", "^", "v", "*"]

    for idx, (m_name, m_info) in enumerate(models_data.items()):
        level_reps = m_info.get("level_reports", {})
        scores = [level_reps.get(lvl, {}).get("pass_at_1", 0.0) * 100 for lvl in levels]
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        plt.plot(levels, scores, marker=m, color=c, linewidth=2.2, label=m_name)

    plt.title("Reduction Ladder: Multi-Arm Performance Comparison", fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Ladder Level", fontsize=11, fontweight="bold")
    plt.ylabel("Pass@1 Accuracy (%)", fontsize=11, fontweight="bold")
    plt.xticks(range(6), level_labels, rotation=10)
    plt.ylim(-2, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    return output_filepath

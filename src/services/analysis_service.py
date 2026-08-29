"""AnalysisService calculating comparative ladder metrics and plotting degradation curves."""

import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd


class AnalysisService:
    """Computes comparative metrics (AUC, Collapse Point, MRI) and generates figures."""

    @staticmethod
    def compute_summary_table(suite_reports: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create structured comparison DataFrame from multiple model evaluation runs."""
        rows = []
        for model_key, rep in suite_reports.items():
            level_reps = rep.get("level_reports", {})
            row = {
                "Model": model_key,
                "L0 (HumanEval)": f"{level_reps.get('L0', {}).get('pass_at_1', 0.0)*100:.1f}%",
                "L1 (Subtle)": f"{level_reps.get('L1', {}).get('pass_at_1', 0.0)*100:.1f}%",
                "L2 (ToolUse)": f"{level_reps.get('L2', {}).get('pass_at_1', 0.0)*100:.1f}%",
                "L3 (Creative)": f"{level_reps.get('L3', {}).get('pass_at_1', 0.0)*100:.1f}%",
                "L4 (Difficult)": f"{level_reps.get('L4', {}).get('pass_at_1', 0.0)*100:.1f}%",
                "L5 (Combine)": f"{level_reps.get('L5', {}).get('pass_at_1', 0.0)*100:.1f}%",
                "Ladder AUC": f"{rep.get('ladder_auc', 0.0)*100:.1f}%",
                "Collapse Point": rep.get("collapse_point", "N/A"),
                "Consistency Delta": f"{rep.get('consistency_delta', 0.0)*100:.1f}%",
            }
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def plot_ladder_curves(
        suite_reports: Dict[str, Dict[str, Any]],
        output_filepath: str = "results/ladder_comparison.png"
    ) -> None:
        """Plot comparative Pass@1 degradation curves across L0-L5."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
        level_labels = ["L0 (HumanEval)", "L1 (Subtle)", "L2 (ToolUse)", "L3 (Creative)", "L4 (Difficult)", "L5 (Combine)"]

        plt.figure(figsize=(10, 6), dpi=300)
        plt.axhline(0.50, color="red", linestyle="--", alpha=0.7, label="Collapse Threshold (50%)")

        for model_key, rep in suite_reports.items():
            level_reps = rep.get("level_reports", {})
            scores = [level_reps.get(lvl, {}).get("pass_at_1", 0.0) for lvl in levels]
            plt.plot(levels, scores, marker="o", linewidth=2.2, label=model_key)

        plt.title("Reduction Ladder for Code: Pass@1 Degradation Across Transformations", fontsize=13, pad=12)
        plt.xlabel("Ladder Level", fontsize=11)
        plt.ylabel("Pass@1 Accuracy", fontsize=11)
        plt.xticks(levels, level_labels, rotation=15)
        plt.ylim(-0.02, 1.02)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, loc="lower left")
        plt.tight_layout()
        plt.savefig(output_filepath)
        plt.close()
        print(f"[PLOT] Saved comparative degradation plot to '{output_filepath}'.")

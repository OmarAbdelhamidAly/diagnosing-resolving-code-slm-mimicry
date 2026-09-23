"""AnalysisService calculating comparative ladder metrics and plotting degradation curves."""

import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd

# Import the evaluation metric stack (no training dependencies)
try:
    from src.evaluation import metrics as _eval_metrics
except ImportError:
    _eval_metrics = None


class AnalysisService:
    """Computes comparative metrics (AUC, Collapse Point, MRI) and generates figures."""

    @staticmethod
    def _normalize_report(rep: Any) -> Dict[str, Any]:
        """Convert ModelEvaluationSuiteReport or dict to normalized dict."""
        if hasattr(rep, "to_dict"):
            return rep.to_dict()
        elif isinstance(rep, dict):
            return rep
        else:
            return vars(rep)

    @staticmethod
    def compute_summary_table(suite_reports: Dict[str, Any]) -> pd.DataFrame:
        """Create structured comparison DataFrame from multiple model evaluation runs."""
        rows = []
        for model_key, raw_rep in suite_reports.items():
            rep = AnalysisService._normalize_report(raw_rep)
            level_reps = rep.get("level_reports", {})

            def _get_p1(lvl_key: str) -> float:
                lvl_val = level_reps.get(lvl_key, {})
                if hasattr(lvl_val, "pass_at_1"):
                    return lvl_val.pass_at_1
                elif isinstance(lvl_val, dict):
                    return lvl_val.get("pass_at_1", 0.0)
                return 0.0

            p1_map = {lvl: _get_p1(lvl) for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]}
            slope = (
                rep.get("degradation_slope")
                if rep.get("degradation_slope") is not None
                else (_eval_metrics.compute_degradation_slope(p1_map) if _eval_metrics else None)
            )
            ctrl_p1 = _get_p1("Ctrl")

            row = {
                "Model": model_key,
                "L0 (HumanEval)": f"{_get_p1('L0')*100:.1f}%",
                "L1 (Subtle)": f"{_get_p1('L1')*100:.1f}%",
                "L2 (ToolUse)": f"{_get_p1('L2')*100:.1f}%",
                "L3 (Creative)": f"{_get_p1('L3')*100:.1f}%",
                "L4 (Difficult)": f"{_get_p1('L4')*100:.1f}%",
                "L5 (Combine)": f"{_get_p1('L5')*100:.1f}%",
                "Ctrl (OOD)": f"{ctrl_p1*100:.1f}%" if ctrl_p1 else "N/A",
                "Ladder AUC": f"{rep.get('ladder_auc', 0.0)*100:.1f}%",
                "Degrad. Slope": f"{slope:+.3f}" if slope is not None else "N/A",
                "Collapse Point": rep.get("collapse_point", "N/A"),
                "Consistency Delta": f"{rep.get('consistency_delta', 0.0)*100:.1f}%",
            }
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def plot_ladder_curves(
        suite_reports: Dict[str, Any],
        output_filepath: str = "results/ladder_comparison.png"
    ) -> None:
        """Plot comparative Pass@1 degradation curves across L0-L5 and Ctrl (OOD)."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        levels = ["L0", "L1", "L2", "L3", "L4", "L5", "Ctrl"]
        level_labels = ["L0 (HumanEval)", "L1 (Subtle)", "L2 (ToolUse)", "L3 (Creative)", "L4 (Difficult)", "L5 (Combine)", "Ctrl (LiveCode)"]

        plt.figure(figsize=(12, 6), dpi=300)
        plt.axhline(0.50, color="red", linestyle="--", alpha=0.7, label="Collapse Threshold (50%)")
        # Shade OOD control zone (Ctrl column)
        plt.axvspan(5.5, 6.5, alpha=0.05, color="purple", label="OOD control zone")

        for model_key, raw_rep in suite_reports.items():
            rep = AnalysisService._normalize_report(raw_rep)
            level_reps = rep.get("level_reports", {})

            def _get_p1(lvl_key: str) -> float:
                lvl_val = level_reps.get(lvl_key, {})
                if hasattr(lvl_val, "pass_at_1"):
                    return lvl_val.pass_at_1
                elif isinstance(lvl_val, dict):
                    return lvl_val.get("pass_at_1", 0.0)
                return 0.0

            scores = [_get_p1(lvl) for lvl in levels]
            plt.plot(range(len(levels)), scores, marker="o", linewidth=2.2, label=model_key)

        plt.title("Reduction Ladder for Code: Pass@1 Degradation Across Transformations", fontsize=13, pad=12)
        plt.xlabel("Ladder Level", fontsize=11)
        plt.ylabel("Pass@1 Accuracy", fontsize=11)
        plt.xticks(range(len(levels)), level_labels, rotation=15)
        plt.ylim(-0.02, 1.02)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, loc="lower left")
        plt.tight_layout()
        plt.savefig(output_filepath)
        plt.close()
        print(f"[PLOT] Saved comparative degradation plot to '{output_filepath}'.")

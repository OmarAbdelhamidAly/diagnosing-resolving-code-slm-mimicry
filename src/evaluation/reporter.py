"""EvaluationReporter: publication-quality comparison plots and tables.

Generates 4 figures and 1 master table from cached eval_report.json files.

Figures
-------
1. 4-curve degradation plot   — Pass@1 across L0→L5→Ctrl for all 4 models
2. Per-model radar chart      — 6-axis metric fingerprint
3. Grouped bar chart          — Pass@1 per level, side-by-side
4. Metric heatmap             — models × metrics colour matrix
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from . import metrics as M


# ── colour palette (print-safe, colour-blind friendly) ───────────────────────
_PALETTE = {
    "M1_baseline":      "#6c757d",   # grey
    "M2_vanilla_sft":   "#0077b6",   # blue
    "M4_standard_grpo": "#f4a261",   # amber
    "M6_inv_grpo":      "#2dc653",   # green
}
_DEFAULT_COLORS = ["#6c757d", "#0077b6", "#f4a261", "#2dc653", "#e63946", "#8338ec"]

_LEVEL_LABELS = {
    "L0": "L0\nHumanEval",
    "L1": "L1\nSubtle",
    "L2": "L2\nToolUse",
    "L3": "L3\nCreative",
    "L4": "L4\nDifficult",
    "L5": "L5\nCombine",
    "Ctrl": "Ctrl\nLiveCode",
}


class EvaluationReporter:
    """Generates comparison figures and master metric table.

    Parameters
    ----------
    reports : dict mapping model_id → eval_report dict (from eval_report.json)
    output_dir : directory where figures will be saved
    """

    def __init__(
        self,
        reports: Dict[str, Dict[str, Any]],
        output_dir: str = "results/figures",
    ) -> None:
        self.reports = reports
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model_ids = list(reports.keys())

    # ------------------------------------------------------------------
    # Helper: extract Pass@1 per level from a report dict
    # ------------------------------------------------------------------

    def _get_p1(self, report: Dict[str, Any], level: str) -> float:
        level_reps = report.get("level_reports", {})
        lvl = level_reps.get(level, {})
        if isinstance(lvl, dict):
            return float(lvl.get("pass_at_1", 0.0))
        if hasattr(lvl, "pass_at_1"):
            return float(lvl.pass_at_1)
        return 0.0

    def _get_color(self, model_id: str, idx: int) -> str:
        return _PALETTE.get(model_id, _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])

    # ------------------------------------------------------------------
    # 1. Master comparison table
    # ------------------------------------------------------------------

    def build_master_table(self) -> pd.DataFrame:
        """Return a formatted DataFrame with all models × all metrics."""
        rows = []
        m1_report = None
        # find M1 for relative gain calculation
        for mid, rep in self.reports.items():
            if "baseline" in mid.lower() or "M1" in mid:
                m1_report = rep
                break

        for mid, rep in self.reports.items():
            p1_map = {
                lvl: self._get_p1(rep, lvl)
                for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]
            }
            auc = rep.get("ladder_auc") or M.compute_ladder_auc(p1_map)
            slope = rep.get("degradation_slope") or M.compute_degradation_slope(p1_map)
            collapse = rep.get("collapse_point", "N/A")
            cons = rep.get("consistency_delta") or M.compute_consistency_delta(
                p1_map.get("L1", 0.0), p1_map.get("L2", 0.0)
            )
            ood = rep.get("ood_score") or self._get_p1(rep, "Ctrl")
            tax = rep.get("overthinking_tax") or 0.0
            rel = (
                M.compute_relative_gain(auc, m1_report.get("ladder_auc", 0.0))
                if m1_report and mid not in ("M1_baseline",)
                else 0.0
            )

            row = {
                "Model": mid,
                "L0": f"{p1_map['L0']*100:.1f}%",
                "L1": f"{p1_map['L1']*100:.1f}%",
                "L2": f"{p1_map['L2']*100:.1f}%",
                "L3": f"{p1_map['L3']*100:.1f}%",
                "L4": f"{p1_map['L4']*100:.1f}%",
                "L5": f"{p1_map['L5']*100:.1f}%",
                "Ctrl (OOD)": f"{ood*100:.1f}%" if ood else "N/A",
                "Ladder AUC": f"{auc*100:.1f}%",
                "Degrad. Slope": f"{slope:+.3f}",
                "Collapse Point": collapse,
                "Consistency Δ": f"{cons*100:.1f}%",
                "Overthinking Tax": f"{tax:.3f}",
                "Rel. Gain vs M1": f"{rel*100:+.1f}%",
            }
            rows.append(row)

        return pd.DataFrame(rows).set_index("Model")

    # ------------------------------------------------------------------
    # 2. 4-curve degradation plot
    # ------------------------------------------------------------------

    def plot_degradation_curves(self, filename: str = "fig1_degradation_curves.png") -> Path:
        """Plot Pass@1 degradation across L0→L5→Ctrl for all models."""
        levels = ["L0", "L1", "L2", "L3", "L4", "L5", "Ctrl"]
        x_labels = [_LEVEL_LABELS[lvl] for lvl in levels]
        x_pos = list(range(len(levels)))

        fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
        ax.axhline(0.50, color="#e63946", linestyle="--", linewidth=1.2,
                   alpha=0.7, label="Collapse threshold (50%)")
        # Shade the Ctrl zone
        ax.axvspan(5.5, 6.5, alpha=0.05, color="#8338ec", label="OOD control zone")

        for idx, (mid, rep) in enumerate(self.reports.items()):
            color = self._get_color(mid, idx)
            scores = [self._get_p1(rep, lvl) for lvl in levels]
            ax.plot(x_pos, scores, marker="o", linewidth=2.3,
                    color=color, label=mid, markersize=6, zorder=3)
            # Fill under curve (subtle)
            ax.fill_between(x_pos, 0, scores, alpha=0.04, color=color)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Ladder Level", fontsize=11)
        ax.set_ylabel("Pass@1 Accuracy", fontsize=11)
        ax.set_title(
            "Reduction Ladder for Code: Pass@1 Degradation Across Transformations",
            fontsize=12, pad=10, fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.legend(frameon=True, loc="lower left", fontsize=9)
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved -> {out}")
        return out

    # ------------------------------------------------------------------
    # 3. Per-model radar chart
    # ------------------------------------------------------------------

    def plot_radar_chart(self, filename: str = "fig2_radar_chart.png") -> Path:
        """6-axis radar chart comparing model metric fingerprints."""
        metric_keys = ["Ladder AUC", "L0", "L5", "Consistency Δ", "Degrad. Slope", "Ctrl (OOD)"]
        n = len(metric_keys)
        angles = [k * 2 * np.pi / n for k in range(n)] + [0]  # close the polygon

        fig, ax = plt.subplots(figsize=(7, 7), dpi=200, subplot_kw=dict(polar=True))

        table = self.build_master_table()

        for idx, mid in enumerate(self._model_ids):
            color = self._get_color(mid, idx)
            row = table.loc[mid] if mid in table.index else None
            if row is None:
                continue

            def _pct(val_str: str) -> float:
                """Convert '72.3%' or '+0.3%' to float 0-1."""
                try:
                    return abs(float(val_str.replace("%", "").replace("+", ""))) / 100.0
                except Exception:
                    return 0.0

            def _slope_norm(val_str: str) -> float:
                """Normalise slope: closer to 0 is better; map [-0.2, 0] → [0, 1]."""
                try:
                    s = float(val_str)
                    return max(0.0, min(1.0, 1.0 + s / 0.2))
                except Exception:
                    return 0.5

            values = [
                _pct(row["Ladder AUC"]),
                _pct(row["L0"]),
                _pct(row["L5"]),
                1.0 - _pct(row["Consistency Δ"]),   # lower delta is better
                _slope_norm(row["Degrad. Slope"]),   # lower |slope| is better
                _pct(row["Ctrl (OOD)"]) if row["Ctrl (OOD)"] != "N/A" else 0.0,
            ]
            values += [values[0]]  # close polygon

            ax.plot(angles, values, linewidth=2, color=color, label=mid)
            ax.fill(angles, values, alpha=0.07, color=color)

        ax.set_thetagrids(
            [a * 180 / np.pi for a in angles[:-1]],
            labels=metric_keys,
            fontsize=9,
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
        ax.set_title("Model Metric Fingerprints (Radar Chart)", fontsize=12,
                     pad=15, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved -> {out}")
        return out

    # ------------------------------------------------------------------
    # 4. Grouped bar chart
    # ------------------------------------------------------------------

    def plot_grouped_bars(self, filename: str = "fig3_grouped_bars.png") -> Path:
        """Grouped bar chart: Pass@1 per level for each model."""
        levels = ["L0", "L1", "L2", "L3", "L4", "L5", "Ctrl"]
        n_levels = len(levels)
        n_models = len(self._model_ids)
        bar_width = 0.7 / n_models
        x = np.arange(n_levels)

        fig, ax = plt.subplots(figsize=(13, 6), dpi=200)
        ax.axhline(0.50, color="#e63946", linestyle="--", linewidth=1.0,
                   alpha=0.6, zorder=1)

        for i, (mid, rep) in enumerate(self.reports.items()):
            color = self._get_color(mid, i)
            offset = (i - n_models / 2.0 + 0.5) * bar_width
            scores = [self._get_p1(rep, lvl) * 100 for lvl in levels]
            bars = ax.bar(x + offset, scores, width=bar_width * 0.92,
                          color=color, label=mid, zorder=2, alpha=0.88)
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                if score > 3:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{score:.0f}",
                        ha="center", va="bottom", fontsize=6, color="dimgrey",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([_LEVEL_LABELS[lvl] for lvl in levels], fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_xlabel("Ladder Level", fontsize=11)
        ax.set_ylabel("Pass@1 (%)", fontsize=11)
        ax.set_title("Pass@1 per Level — Model Comparison", fontsize=12,
                     pad=10, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved -> {out}")
        return out

    # ------------------------------------------------------------------
    # 5. Metric heatmap
    # ------------------------------------------------------------------

    def plot_metric_heatmap(self, filename: str = "fig4_metric_heatmap.png") -> Path:
        """Heatmap of numeric metrics across models."""
        table = self.build_master_table()

        # Select numeric columns only
        numeric_cols = ["L0", "L1", "L2", "L3", "L4", "L5",
                        "Ladder AUC", "Consistency Δ", "Ctrl (OOD)"]
        numeric_cols = [c for c in numeric_cols if c in table.columns]

        def _to_float(s: str) -> float:
            try:
                return float(str(s).replace("%", "").replace("+", "").strip())
            except Exception:
                return float("nan")

        _map_fn = getattr(table[numeric_cols], "map", getattr(table[numeric_cols], "applymap", None))
        matrix = _map_fn(_to_float).values.astype(float)
        row_labels = list(table.index)

        fig, ax = plt.subplots(figsize=(len(numeric_cols) * 1.4 + 1, len(row_labels) * 0.9 + 1.5), dpi=200)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label="Score (%)", shrink=0.7)

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)

        for i in range(len(row_labels)):
            for j in range(len(numeric_cols)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=8, color="black" if 30 < val < 80 else "white")

        ax.set_title("Metric Heatmap: Models × Metrics", fontsize=12,
                     pad=10, fontweight="bold")
        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved -> {out}")
        return out

    # ------------------------------------------------------------------
    # Generate all 4 figures + table in one call
    # ------------------------------------------------------------------

    def generate_all(self) -> Dict[str, Any]:
        """Generate all 4 figures and return the master table + figure paths."""
        print("\n[Reporter] Generating publication figures...")
        table = self.build_master_table()
        p1 = self.plot_degradation_curves()
        p2 = self.plot_radar_chart()
        p3 = self.plot_grouped_bars()
        p4 = self.plot_metric_heatmap()
        print("[Reporter] All figures saved.\n")
        return {
            "master_table": table,
            "fig1_degradation": str(p1),
            "fig2_radar": str(p2),
            "fig3_bars": str(p3),
            "fig4_heatmap": str(p4),
        }

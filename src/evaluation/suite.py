"""EvaluationSuite: model-agnostic orchestrator for the full evaluation pipeline.

This class is the single entry-point for evaluating any checkpoint against
the fixed Reduction Ladder benchmark pool.  It intentionally has NO dependency
on any training code and must remain that way.

Usage
-----
    from src.evaluation import EvaluationSuite, BenchmarkRegistry

    registry = BenchmarkRegistry()
    suite = EvaluationSuite(registry)

    # Evaluate the raw baseline (no adapter)
    report = suite.evaluate_model(
        model_id="M1_baseline",
        model_name_or_path="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        adapter_path=None,
        output_dir="results/M1_baseline",
    )

    # Evaluate a LoRA-trained checkpoint
    report = suite.evaluate_model(
        model_id="M6_inv_grpo",
        model_name_or_path="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        adapter_path="checkpoints/inv_grpo_final",
        output_dir="results/M6_inv_grpo",
    )
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.entities import (
    BenchmarkTask,
    LevelEvaluationReport,
    ModelEvaluationSuiteReport,
)
from src.infrastructure.sandbox import MultiprocessSandbox
from src.infrastructure.classifier import RuleBasedErrorClassifier

from .registry import BenchmarkRegistry, LEVEL_ORDER
from . import metrics as M


class EvaluationSuite:
    """Model-agnostic evaluation orchestrator.

    Parameters
    ----------
    registry : BenchmarkRegistry instance (pre-loaded benchmark pool)
    executor : ICodeExecutor-compatible object; defaults to MultiprocessSandbox
    classifier : IErrorClassifier-compatible object; defaults to RuleBasedErrorClassifier
    evaluate_pass5 : whether to also run Pass@5 (5 sampled completions per task)
    timeout_seconds : per-task sandbox execution timeout
    """

    def __init__(
        self,
        registry: BenchmarkRegistry,
        executor=None,
        classifier=None,
        evaluate_pass5: bool = True,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.registry = registry
        self.executor = executor or MultiprocessSandbox()
        self.classifier = classifier or RuleBasedErrorClassifier()
        self.evaluate_pass5 = evaluate_pass5
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model_id: str,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        output_dir: str = "results",
        levels: Optional[List[str]] = None,
    ) -> ModelEvaluationSuiteReport:
        """Run the full benchmark suite for one model checkpoint.

        Parameters
        ----------
        model_id        : short identifier used in filenames, e.g. "M6_inv_grpo"
        model_name_or_path : HuggingFace model name or local path
        adapter_path    : path to a PEFT LoRA adapter directory, or None for base model
        output_dir      : directory where eval_report.json will be saved
        levels          : subset of levels to run (default: all available)

        Returns
        -------
        ModelEvaluationSuiteReport with all metric fields populated
        """
        print(f"\n{'='*60}")
        print(f"[EvaluationSuite] Evaluating: {model_id}")
        print(f"  Base model  : {model_name_or_path}")
        print(f"  Adapter     : {adapter_path or 'None (base model)'}")
        print(f"  Pass@5      : {self.evaluate_pass5}")
        print(f"{'='*60}\n")

        # Lazy import — avoids importing torch/transformers when just loading metrics
        runner = self._build_runner(model_name_or_path, adapter_path)

        levels_to_run = levels or self.registry.available_levels()
        suite_report = ModelEvaluationSuiteReport(
            model_name=model_id,
            checkpoint_path=adapter_path or model_name_or_path,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

        # ── per-level evaluation ──────────────────────────────────────
        for level_key in LEVEL_ORDER:
            if level_key not in levels_to_run:
                continue
            tasks = self.registry.get_tasks(level_key)
            if not tasks:
                print(f"[SKIP] {level_key}: no tasks available")
                continue
            level_report = self._evaluate_level(runner, tasks, level_key)
            suite_report.level_reports[level_key] = level_report

        # ── compute suite-level metrics ───────────────────────────────
        self._compute_suite_metrics(suite_report)

        # ── persist ──────────────────────────────────────────────────
        out_path = Path(output_dir) / model_id / "eval_report.json"
        self._save_report(suite_report, out_path)

        print(f"\n[EvaluationSuite] Done → {out_path}")
        return suite_report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_runner(self, model_name_or_path: str, adapter_path: Optional[str]):
        """Dynamically build the model runner (avoids top-level torch import)."""
        from src.infrastructure.model_runner import HuggingFaceModelRunner
        return HuggingFaceModelRunner(
            model_name_or_path=model_name_or_path,
            adapter_path=adapter_path,
        )

    def _evaluate_level(
        self,
        runner,
        tasks: List[BenchmarkTask],
        level_key: str,
    ) -> LevelEvaluationReport:
        """Evaluate one ladder level and return a LevelEvaluationReport."""
        from tqdm import tqdm

        print(f"\n[LEVEL] {level_key}  ({len(tasks)} tasks)")
        passed_p1 = 0
        p5_pass_counts: List[int] = []
        error_counts: Dict[str, int] = {}
        task_records = []
        total_tokens = 0.0

        for task in tqdm(tasks, desc=f"  {level_key}", leave=False):
            # ── Greedy Pass@1 ─────────────────────────────────────────
            p1_completions = runner.generate(
                prompt=task.prompt,
                temperature=0.0,
                num_samples=1,
                max_new_tokens=1024,
            )
            p1_code = p1_completions[0] if p1_completions else ""
            p1_exec = self.executor.execute(
                prompt=task.prompt,
                solution=p1_code,
                test=task.test,
                entry_point=task.entry_point,
                timeout_seconds=self.timeout_seconds,
            )
            p1_category = self.classifier.classify(task, p1_code, p1_exec)
            error_counts[p1_category] = error_counts.get(p1_category, 0) + 1
            if p1_exec.passed:
                passed_p1 += 1

            token_len = len(p1_code.split())
            total_tokens += token_len

            record: Dict[str, Any] = {
                "task_id": task.task_id,
                "p1_passed": p1_exec.passed,
                "p1_status": p1_exec.status,
                "p1_error_category": p1_category,
                "p1_code": p1_code,
                "p1_error_message": p1_exec.error_message,
                "p1_time": p1_exec.execution_time_seconds,
                "approx_tokens": token_len,
            }

            # ── Sampled Pass@5 ────────────────────────────────────────
            if self.evaluate_pass5:
                p5_completions = runner.generate(
                    prompt=task.prompt,
                    temperature=0.8,
                    num_samples=5,
                    max_new_tokens=1024,
                )
                n_passed = 0
                p5_samples = []
                for idx, code in enumerate(p5_completions):
                    res = self.executor.execute(
                        prompt=task.prompt,
                        solution=code,
                        test=task.test,
                        entry_point=task.entry_point,
                        timeout_seconds=self.timeout_seconds,
                    )
                    if res.passed:
                        n_passed += 1
                    p5_samples.append({"sample_idx": idx, "passed": res.passed, "status": res.status})
                p5_pass_counts.append(n_passed)
                record["p5_n_passed"] = n_passed
                record["p5_samples"] = p5_samples

            task_records.append(record)

        total = len(tasks)
        pass_at_1 = passed_p1 / total if total > 0 else 0.0

        # Compute unbiased Pass@5 using Chen et al. estimator
        pass_at_5 = None
        if self.evaluate_pass5 and p5_pass_counts:
            pass_at_5 = sum(
                M.compute_pass_at_k(n=5, c=c, k=5) for c in p5_pass_counts
            ) / len(p5_pass_counts)

        avg_tokens = total_tokens / total if total > 0 else 0.0

        print(
            f"  ✓ Pass@1={pass_at_1*100:.1f}%"
            + (f"  Pass@5={pass_at_5*100:.1f}%" if pass_at_5 is not None else "")
            + f"  Avg tokens≈{avg_tokens:.0f}"
        )

        return LevelEvaluationReport(
            ladder_level=level_key,
            benchmark_name=tasks[0].benchmark if tasks else level_key,
            total_tasks=total,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            error_breakdown=error_counts,
            average_token_length=avg_tokens,
            task_results=task_records,
        )

    def _compute_suite_metrics(self, suite: ModelEvaluationSuiteReport) -> None:
        """Populate all suite-level metric fields from level reports."""
        def _p1(lvl: str) -> Optional[float]:
            rep = suite.level_reports.get(lvl)
            if rep is None:
                return None
            return rep.pass_at_1 if hasattr(rep, "pass_at_1") else rep.get("pass_at_1")

        p1_map = {lvl: _p1(lvl) for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"] if _p1(lvl) is not None}

        suite.ladder_auc = M.compute_ladder_auc(p1_map)
        suite.collapse_point = M.compute_collapse_point(p1_map)
        suite.consistency_delta = M.compute_consistency_delta(
            p1_map.get("L1", 0.0), p1_map.get("L2", 0.0)
        )

        # Extra metrics stored in the to_dict() but not in the dataclass fields
        # We attach them as dynamic attributes so the reporter can access them
        suite._degradation_slope = M.compute_degradation_slope(p1_map)   # type: ignore[attr-defined]
        suite._ood_score = _p1("Ctrl") if _p1("Ctrl") is not None else None  # type: ignore[attr-defined]

        # Overthinking Tax (uses average token length across all L0-L5 levels)
        all_avg_tokens = []
        for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]:
            rep = suite.level_reports.get(lvl)
            if rep is not None and hasattr(rep, "average_token_length"):
                all_avg_tokens.append(rep.average_token_length)
        avg_tok = sum(all_avg_tokens) / len(all_avg_tokens) if all_avg_tokens else 0.0
        suite._overthinking_tax = M.compute_overthinking_tax(suite.ladder_auc, avg_tok)  # type: ignore[attr-defined]

    def _save_report(self, suite: ModelEvaluationSuiteReport, path: Path) -> None:
        """Serialize and save the evaluation report to a JSON file."""
        os.makedirs(path.parent, exist_ok=True)
        data = suite.to_dict()
        # Attach extra metrics computed in _compute_suite_metrics
        data["degradation_slope"] = getattr(suite, "_degradation_slope", None)
        data["ood_score"] = getattr(suite, "_ood_score", None)
        data["overthinking_tax"] = getattr(suite, "_overthinking_tax", None)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        print(f"  [SAVE] eval_report.json → {path}")

    @staticmethod
    def load_report(json_path: str | Path) -> Dict[str, Any]:
        """Load a previously saved eval_report.json as a plain dict."""
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

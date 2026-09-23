"""Unified EvaluationEngine orchestrating the complete Reduction Ladder evaluation pipeline.

Provides a clean, one-stop interface for evaluating any model checkpoint
across L0-L5 benchmarks, diagnosing error taxonomies, computing metrics,
and generating publication-grade plots and JSON reports.
"""

import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm

from src.core.entities import BenchmarkTask, ExecutionResult, LevelEvaluationReport, ModelEvaluationSuiteReport
from src.core.interfaces import IModelRunner, ICodeExecutor, IErrorClassifier
from src.infrastructure.hf_loader import HuggingFaceBenchmarkLoader
from src.infrastructure.model_loader import QuantizedModelRunner
from src.infrastructure.sandbox import SubprocessSandbox
from src.infrastructure.classifier import RuleBasedErrorClassifier
from src.infrastructure.persistence import save_json, load_json
from src.evaluation.metrics import (
    compute_ladder_auc,
    compute_collapse_point,
    compute_consistency_delta,
    compute_mri_v2 as compute_mri,
    compute_overthinking_tax as compute_token_efficiency,
)
from src.shared.plots import (
    plot_single_model_degradation,
    plot_error_taxonomy,
    plot_multi_model_comparison,
)
from src.core.config import settings


class EvaluationEngine:
    """Consolidated, high-level evaluation engine for Reduction Ladder research."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
        model_runner: Optional[IModelRunner] = None,
        executor: Optional[ICodeExecutor] = None,
        classifier: Optional[IErrorClassifier] = None,
        data_cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        default_timeout: Optional[float] = None,
    ):
        self.model_name = model_name or settings.models.student_model
        self.adapter_path = adapter_path
        self.data_cache_dir = data_cache_dir or settings.storage.ladder_cache_dir
        self.results_dir = results_dir or settings.storage.results_dir
        self.timeout = default_timeout or settings.evaluation.timeout_seconds

        # Initialize components lazily or with provided instances
        self.loader = HuggingFaceBenchmarkLoader(cache_dir=self.data_cache_dir)
        self.executor = executor or SubprocessSandbox(default_timeout=self.timeout)
        self.classifier = classifier or RuleBasedErrorClassifier()

        if model_runner is not None:
            self.model_runner = model_runner
        else:
            self.model_runner = QuantizedModelRunner(
                model_name_or_path=self.model_name,
                adapter_path=adapter_path
            )

        self.ladder_data: Dict[str, List[BenchmarkTask]] = {}
        self.latest_suite_report: Optional[ModelEvaluationSuiteReport] = None

    def load_benchmarks(self) -> Dict[str, List[BenchmarkTask]]:
        """Load all L0-L5 ladder benchmark levels from local cache."""
        self.ladder_data = self.loader.load_all_levels(force_download=False)
        return self.ladder_data

    def evaluate_level(
        self,
        tasks: List[BenchmarkTask],
        level_name: str,
        evaluate_pass5: bool = False,
        timeout_seconds: float = 5.0,
        batch_size: int = 8,
    ) -> LevelEvaluationReport:
        """Run evaluation on a single ladder level using batched GPU inference
        and parallel subprocess execution."""
        print(f"\n[EVAL] Evaluating Level: {level_name} ({len(tasks)} tasks)...")

        # ── Phase 1: Batched greedy generation (Pass@1) ──────────────────────
        print(f"  [GEN]  Generating Pass@1 completions in batches of {batch_size}...")
        p1_codes: List[str] = []
        prompts = [t.prompt for t in tasks]
        for batch_start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"{level_name} batches",
            unit="batch",
        ):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            batch_results = self.model_runner.generate_batch(
                prompts=batch_prompts,
                temperature=0.0,
                num_samples=1,
                max_new_tokens=1024,
            )
            p1_codes.extend(r[0] if r else "" for r in batch_results)

        # ── Phase 2: Parallel Pass@5 generation (optional) ───────────────────
        p5_codes: List[List[str]] = [[] for _ in tasks]
        if evaluate_pass5:
            print(f"  [GEN]  Generating Pass@5 samples in batches of {batch_size}...")
            for batch_start in tqdm(
                range(0, len(prompts), batch_size),
                desc=f"{level_name} P@5 batches",
                unit="batch",
            ):
                batch_prompts = prompts[batch_start : batch_start + batch_size]
                batch_results = self.model_runner.generate_batch(
                    prompts=batch_prompts,
                    temperature=0.8,
                    num_samples=5,
                    max_new_tokens=1024,
                )
                for j, samples in enumerate(batch_results):
                    p5_codes[batch_start + j] = samples

        # ── Phase 3: Parallel sandbox execution ───────────────────────────────
        print(f"  [EXEC] Running sandbox execution in parallel...")

        def _run_p1(idx: int):
            task = tasks[idx]
            code = p1_codes[idx]
            result = self.executor.execute(
                prompt=task.prompt,
                solution=code,
                test=task.test,
                entry_point=task.entry_point,
                timeout_seconds=timeout_seconds,
            )
            category = self.classifier.classify(task, code, result)
            return idx, result, category

        p1_exec_results: List[Optional[ExecutionResult]] = [None] * len(tasks)
        p1_categories: List[str] = [""] * len(tasks)
        max_workers = min(16, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_p1, i): i for i in range(len(tasks))}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"{level_name} exec"):
                idx, result, category = fut.result()
                p1_exec_results[idx] = result
                p1_categories[idx] = category

        # ── Phase 4: Assemble records & metrics ───────────────────────────────
        passed_p1 = 0
        passed_p5_candidates = 0
        error_counts: Dict[str, int] = {}
        task_records = []

        for idx, task in enumerate(tasks):
            p1_exec = p1_exec_results[idx]
            p1_category = p1_categories[idx]
            error_counts[p1_category] = error_counts.get(p1_category, 0) + 1
            if p1_exec.passed:
                passed_p1 += 1

            record = {
                "task_id": task.task_id,
                "p1_passed": p1_exec.passed,
                "p1_status": p1_exec.status,
                "p1_error_category": p1_category,
                "p1_code": p1_codes[idx],
                "p1_error_message": p1_exec.error_message,
                "p1_time": p1_exec.execution_time_seconds,
            }

            if evaluate_pass5 and p5_codes[idx]:
                any_p5_passed = False
                p5_results_rec = []
                for s_idx, c in enumerate(p5_codes[idx]):
                    res = self.executor.execute(
                        prompt=task.prompt,
                        solution=c,
                        test=task.test,
                        entry_point=task.entry_point,
                        timeout_seconds=timeout_seconds,
                    )
                    if res.passed:
                        any_p5_passed = True
                    p5_results_rec.append({"sample_idx": s_idx, "passed": res.passed, "status": res.status})
                if any_p5_passed:
                    passed_p5_candidates += 1
                record["p5_any_passed"] = any_p5_passed
                record["p5_samples"] = p5_results_rec

            task_records.append(record)

        total = len(tasks)
        pass_at_1 = (passed_p1 / total) if total > 0 else 0.0
        pass_at_5 = (passed_p5_candidates / total) if (evaluate_pass5 and total > 0) else None

        print(f"[OK] {level_name} Result: Pass@1 = {pass_at_1*100:.2f}%" + (f", Pass@5 = {pass_at_5*100:.2f}%" if pass_at_5 is not None else ""))
        print(f"     Error Breakdown: {error_counts}")

        return LevelEvaluationReport(
            ladder_level=level_name,
            benchmark_name=tasks[0].benchmark if tasks else "",
            total_tasks=total,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            error_breakdown=error_counts,
            task_results=task_records
        )

    def run_full_ladder(
        self,
        output_tag: str = "baseline",
        evaluate_pass5: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> ModelEvaluationSuiteReport:
        """Run full evaluation suite across L0-L5 and compute all 5 metric axes."""
        if not self.ladder_data:
            self.load_benchmarks()

        timeout = timeout_seconds or self.timeout
        model_display_name = f"{self.model_name}" + (f" (+Adapter: {os.path.basename(self.adapter_path)})" if self.adapter_path else "")
        suite_report = ModelEvaluationSuiteReport(
            model_name=model_display_name,
            checkpoint_path=self.adapter_path,
            timestamp=datetime.datetime.utcnow().isoformat()
        )

        for level_key in ["L0", "L1", "L2", "L3", "L4", "L5"]:
            if level_key in self.ladder_data:
                rep = self.evaluate_level(
                    tasks=self.ladder_data[level_key],
                    level_name=level_key,
                    evaluate_pass5=evaluate_pass5,
                    timeout_seconds=timeout,
                )
                suite_report.level_reports[level_key] = rep

        # Compute Mathematical Metrics
        p1_dict = {lvl: rep.pass_at_1 for lvl, rep in suite_report.level_reports.items()}
        suite_report.ladder_auc = compute_ladder_auc(p1_dict)
        suite_report.collapse_point = compute_collapse_point(p1_dict, threshold=0.50)

        p1_l1 = p1_dict.get("L1", 0.0)
        p1_l2 = p1_dict.get("L2", 0.0)
        suite_report.consistency_delta = compute_consistency_delta(p1_l1, p1_l2)

        p1_l0 = p1_dict.get("L0", 0.0)
        p1_l3 = p1_dict.get("L3", 0.0)
        suite_report.memorization_risk_index = compute_mri(p1_l0, p1_l3)

        self.latest_suite_report = suite_report

        # Save Report & Plots cleanly in results/<output_tag>/
        model_out_dir = os.path.join(self.results_dir, output_tag)
        os.makedirs(model_out_dir, exist_ok=True)
        report_file = os.path.join(model_out_dir, f"suite_report_{output_tag}.json")

        serializable = {
            "model_name": suite_report.model_name,
            "checkpoint_path": suite_report.checkpoint_path,
            "timestamp": suite_report.timestamp,
            "ladder_auc": suite_report.ladder_auc,
            "collapse_point": suite_report.collapse_point,
            "consistency_delta": suite_report.consistency_delta,
            "memorization_risk_index": suite_report.memorization_risk_index,
            "level_reports": {
                k: {
                    "level": v.ladder_level,
                    "benchmark": v.benchmark_name,
                    "total_tasks": v.total_tasks,
                    "pass_at_1": v.pass_at_1,
                    "pass_at_5": v.pass_at_5,
                    "error_breakdown": v.error_breakdown,
                    "task_results": v.task_results
                }
                for k, v in suite_report.level_reports.items()
            }
        }
        save_json(serializable, report_file)
        print(f"\n[SAVE] Full evaluation report saved to '{report_file}'")

        # Automatically generate plots
        plot_single_model_degradation(
            level_reports=serializable["level_reports"],
            model_name=model_display_name,
            output_filepath=os.path.join(model_out_dir, f"{output_tag}_degradation_curve.png")
        )
        plot_error_taxonomy(
            level_reports=serializable["level_reports"],
            model_name=model_display_name,
            output_filepath=os.path.join(model_out_dir, f"{output_tag}_error_taxonomy.png")
        )

        return suite_report

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return structured summary table of the latest evaluation run."""
        if not self.latest_suite_report:
            raise ValueError("No evaluation has been run yet. Call run_full_ladder() first.")

        rep = self.latest_suite_report
        row = {
            "Model": rep.model_name,
            "L0 (HumanEval)": f"{rep.level_reports.get('L0', LevelEvaluationReport('', '', 0, 0.0)).pass_at_1*100:.1f}%",
            "L1 (Subtle)": f"{rep.level_reports.get('L1', LevelEvaluationReport('', '', 0, 0.0)).pass_at_1*100:.1f}%",
            "L2 (ToolUse)": f"{rep.level_reports.get('L2', LevelEvaluationReport('', '', 0, 0.0)).pass_at_1*100:.1f}%",
            "L3 (Creative)": f"{rep.level_reports.get('L3', LevelEvaluationReport('', '', 0, 0.0)).pass_at_1*100:.1f}%",
            "L4 (Difficult)": f"{rep.level_reports.get('L4', LevelEvaluationReport('', '', 0, 0.0)).pass_at_1*100:.1f}%",
            "L5 (Combine)": f"{rep.level_reports.get('L5', LevelEvaluationReport('', '', 0, 0.0)).pass_at_1*100:.1f}%",
            "Ladder AUC (A)": f"{rep.ladder_auc*100:.1f}%",
            "Collapse Point (l*)": rep.collapse_point,
            "Consistency Delta": f"{rep.consistency_delta*100:.1f}%",
            "MRI Score": f"{rep.memorization_risk_index*100:.1f}%",
        }
        return pd.DataFrame([row])

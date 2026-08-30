"""Unified EvaluationEngine orchestrating the complete Reduction Ladder evaluation pipeline.

Provides a clean, one-stop interface for evaluating any model checkpoint
across L0-L5 benchmarks, diagnosing error taxonomies, computing metrics,
and generating publication-grade plots and JSON reports.
"""

import os
import datetime
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
    compute_mri,
    compute_token_efficiency,
)
from src.evaluation.plots import (
    plot_single_model_degradation,
    plot_error_taxonomy,
    plot_multi_model_comparison,
)


class EvaluationEngine:
    """Consolidated, high-level evaluation engine for Reduction Ladder research."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        model_runner: Optional[IModelRunner] = None,
        executor: Optional[ICodeExecutor] = None,
        classifier: Optional[IErrorClassifier] = None,
        data_cache_dir: str = "data/ladder",
        results_dir: str = "results",
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.data_cache_dir = data_cache_dir
        self.results_dir = results_dir

        # Initialize components lazily or with provided instances
        self.loader = HuggingFaceBenchmarkLoader(cache_dir=data_cache_dir)
        self.executor = executor or SubprocessSandbox(default_timeout=5.0)
        self.classifier = classifier or RuleBasedErrorClassifier()

        if model_runner is not None:
            self.model_runner = model_runner
        else:
            self.model_runner = QuantizedModelRunner(
                model_name_or_path=model_name,
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
        timeout_seconds: float = 5.0
    ) -> LevelEvaluationReport:
        """Run evaluation on a single ladder level."""
        print(f"\n[EVAL] Evaluating Level: {level_name} ({len(tasks)} tasks)...")
        passed_p1 = 0
        passed_p5_candidates = 0
        error_counts: Dict[str, int] = {}
        task_records = []

        for task in tqdm(tasks, desc=f"Evaluating {level_name}"):
            # 1. Greedy Pass@1
            p1_completions = self.model_runner.generate(
                prompt=task.prompt,
                temperature=0.0,
                num_samples=1,
                max_new_tokens=1024
            )
            p1_code = p1_completions[0] if p1_completions else ""
            p1_exec: ExecutionResult = self.executor.execute(
                prompt=task.prompt,
                solution=p1_code,
                test=task.test,
                entry_point=task.entry_point,
                timeout_seconds=timeout_seconds
            )

            p1_category = self.classifier.classify(task, p1_code, p1_exec)
            error_counts[p1_category] = error_counts.get(p1_category, 0) + 1

            if p1_exec.passed:
                passed_p1 += 1

            record = {
                "task_id": task.task_id,
                "p1_passed": p1_exec.passed,
                "p1_status": p1_exec.status,
                "p1_error_category": p1_category,
                "p1_code": p1_code,
                "p1_error_message": p1_exec.error_message,
                "p1_time": p1_exec.execution_time_seconds,
            }

            # 2. Sampled Pass@5 (Optional)
            if evaluate_pass5:
                p5_completions = self.model_runner.generate(
                    prompt=task.prompt,
                    temperature=0.8,
                    num_samples=5,
                    max_new_tokens=1024
                )
                any_p5_passed = False
                p5_results = []
                for idx, c in enumerate(p5_completions):
                    res = self.executor.execute(
                        prompt=task.prompt,
                        solution=c,
                        test=task.test,
                        entry_point=task.entry_point,
                        timeout_seconds=timeout_seconds
                    )
                    if res.passed:
                        any_p5_passed = True
                    p5_results.append({"sample_idx": idx, "passed": res.passed, "status": res.status})

                if any_p5_passed:
                    passed_p5_candidates += 1
                record["p5_any_passed"] = any_p5_passed
                record["p5_samples"] = p5_results

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
        evaluate_pass5: bool = False
    ) -> ModelEvaluationSuiteReport:
        """Run full evaluation suite across L0-L5 and compute all 5 metric axes."""
        if not self.ladder_data:
            self.load_benchmarks()

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
                    evaluate_pass5=evaluate_pass5
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
        report_file = os.path.join(model_out_dir, f"{output_tag}_evaluation_report.json")

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

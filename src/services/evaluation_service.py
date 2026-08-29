"""EvaluationService orchestrating model inference, sandbox execution, and error diagnosis."""

import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from src.core.interfaces import IModelRunner, ICodeExecutor, IErrorClassifier
from src.core.entities import (
    BenchmarkTask,
    ModelSample,
    LevelEvaluationReport,
    ModelEvaluationSuiteReport,
    ExecutionResult,
)
from src.infrastructure.sandbox import MultiprocessSandbox
from src.infrastructure.classifier import RuleBasedErrorClassifier
from src.infrastructure.persistence import save_json


class EvaluationService:
    """Use-case service for evaluating LLMs on the Reduction Ladder."""

    def __init__(
        self,
        model_runner: IModelRunner,
        executor: ICodeExecutor = None,
        classifier: IErrorClassifier = None
    ):
        self.runner = model_runner
        self.executor = executor or MultiprocessSandbox()
        self.classifier = classifier or RuleBasedErrorClassifier()

    def evaluate_level(
        self,
        tasks: List[BenchmarkTask],
        level_name: str,
        evaluate_pass5: bool = False,
        timeout_seconds: float = 5.0
    ) -> LevelEvaluationReport:
        """Run full evaluation on a specific ladder level."""
        print(f"\n[EVAL] Evaluating level: {level_name} ({len(tasks)} tasks)...")
        passed_p1 = 0
        passed_p5_candidates = 0
        error_counts: Dict[str, int] = {}
        task_records = []

        for task in tqdm(tasks, desc=f"Evaluating {level_name}"):
            # 1. Greedy Pass@1 evaluation
            p1_completions = self.runner.generate(
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

            # 2. Sampled Pass@5 evaluation (optional)
            if evaluate_pass5:
                p5_completions = self.runner.generate(
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

        print(f"[EVAL] {level_name} Result: Pass@1 = {pass_at_1*100:.2f}%" + (f", Pass@5 = {pass_at_5*100:.2f}%" if pass_at_5 is not None else ""))
        print(f"[EVAL] Error Breakdown: {error_counts}")

        return LevelEvaluationReport(
            ladder_level=level_name,
            benchmark_name=tasks[0].benchmark if tasks else "",
            total_tasks=total,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            error_breakdown=error_counts,
            task_results=task_records
        )

    def evaluate_suite(
        self,
        ladder_data: Dict[str, List[BenchmarkTask]],
        model_name: str = "Qwen2.5-Coder-1.5B-Instruct",
        checkpoint_path: Optional[str] = None,
        evaluate_pass5: bool = False,
        output_file: Optional[str] = None
    ) -> ModelEvaluationSuiteReport:
        """Run full evaluation suite across all ladder levels (L0 to L5)."""
        suite_report = ModelEvaluationSuiteReport(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            timestamp=datetime.datetime.utcnow().isoformat()
        )

        for level_key in ["L0", "L1", "L2", "L3", "L4", "L5"]:
            if level_key in ladder_data:
                report = self.evaluate_level(
                    tasks=ladder_data[level_key],
                    level_name=level_key,
                    evaluate_pass5=evaluate_pass5
                )
                suite_report.level_reports[level_key] = report

        # Calculate Ladder AUC and Collapse Point
        p1_scores = [suite_report.level_reports[lvl].pass_at_1 for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"] if lvl in suite_report.level_reports]
        if p1_scores:
            suite_report.ladder_auc = sum(p1_scores) / len(p1_scores)

        # Collapse Point: first level where pass@1 < 0.50
        collapse_point = None
        for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]:
            if lvl in suite_report.level_reports:
                if suite_report.level_reports[lvl].pass_at_1 < 0.50:
                    collapse_point = lvl
                    break
        suite_report.collapse_point = collapse_point or "None (Maintained > 50%)"

        # Consistency Delta: |Pass@1(L1) - Pass@1(L2)|
        if "L1" in suite_report.level_reports and "L2" in suite_report.level_reports:
            suite_report.consistency_delta = abs(
                suite_report.level_reports["L1"].pass_at_1 - suite_report.level_reports["L2"].pass_at_1
            )

        if output_file:
            serializable_report = {
                "model_name": suite_report.model_name,
                "checkpoint_path": suite_report.checkpoint_path,
                "timestamp": suite_report.timestamp,
                "ladder_auc": suite_report.ladder_auc,
                "collapse_point": suite_report.collapse_point,
                "consistency_delta": suite_report.consistency_delta,
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
            save_json(serializable_report, output_file)
            print(f"[SAVE] Full evaluation report saved to '{output_file}'.")

        return suite_report

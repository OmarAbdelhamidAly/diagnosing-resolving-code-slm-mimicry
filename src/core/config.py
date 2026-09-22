"""Centralized Configuration Management System for Reduction Ladder.

Follows Clean Architecture principles:
- Single Source of Truth via `config.yaml`
- Environment Variable & `.env` overrides (via python-dotenv & Pydantic v2)
- Zero hardcoding across services, runners, and CLI scripts
- Auto-configuration of HuggingFace cache & storage directories
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

# Automatically discover and load .env file from project root or parents
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)
else:
    load_dotenv(override=False)


def _detect_default_hf_cache() -> str:
    """Intelligently detect drive with ample storage on Windows or fallback to standard."""
    env_cache = os.getenv("HF_CACHE_DIR") or os.getenv("HF_HOME")
    if env_cache:
        return env_cache
    if os.path.exists("D:\\"):
        return "D:/hf_cache"
    elif os.path.exists("E:\\"):
        return "E:/hf_cache"
    return os.path.expanduser("~/.cache/huggingface")


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = "diagnosing-resolving-code-slm-mimicry"
    authors: List[str] = Field(default_factory=lambda: ["Omar Abdelhamid", "Nour Walid"])
    supervisor: str = "Dr. Ghada Soliman"
    institution: str = "Orange Innovation Labs"
    seed: int = 42


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    hf_cache_dir: str = Field(default_factory=_detect_default_hf_cache)
    ladder_cache_dir: str = "data/ladder"
    distillation_cache_dir: str = "data/distillation"
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"

    def ensure_dirs(self, base_path: Optional[Path] = None) -> None:
        """Create storage directories if they do not exist."""
        base = base_path or ROOT_DIR
        for d in [self.ladder_cache_dir, self.distillation_cache_dir, self.results_dir, self.checkpoints_dir]:
            p = Path(d)
            if not p.is_absolute():
                p = base / p
            p.mkdir(parents=True, exist_ok=True)
        # Also ensure HF cache dir exists
        try:
            Path(self.hf_cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    student_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    comparison_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    quantization: str = "4bit_nf4"
    max_seq_len: int = 2048
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"


class BenchmarksConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    l0_humaneval: str = "openai/openai_humaneval"
    l1_subtle: str = "evoeval/EvoEval_subtle"
    l2_verbose: str = "evoeval/EvoEval_tool_use"
    l3_creative: str = "evoeval/EvoEval_creative"
    l4_difficult: str = "evoeval/EvoEval_difficult"
    l5_combine: str = "evoeval/EvoEval_combine"
    control_lcb: str = "livecodebench/code_generation_lite"


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pass_k: List[int] = Field(default_factory=lambda: [1, 5])
    temperature_greedy: float = 0.0
    temperature_sample: float = 0.8
    num_samples: int = 5
    timeout_seconds: float = 5.0
    max_new_tokens: int = 1024


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    num_samples: int = 10000
    cot_source: str = "opencoder_reasoning"
    min_tokens: int = 64
    max_tokens: int = 1800
    vanilla_file: str = "sft_positive_cot.jsonl"
    contrastive_file: str = "sft_contrastive_pairs.jsonl"


class QLoRAConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    warmup_ratio: float = 0.03
    vanilla_output_dir: str = "checkpoints/qlora_vanilla_adapter"
    contrastive_output_dir: str = "checkpoints/qlora_contrastive_adapter"


class InvGRPOConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    group_size: int = 4
    kl_coef: float = 0.1
    lambda_consistency: float = 0.5
    gamma_template_penalty: float = 0.5
    training_steps: int = 1000
    eval_every_steps: int = 100
    batch_size: int = 1
    learning_rate: float = 1.0e-5
    output_dir: str = "checkpoints/inv_grpo_final"


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    benchmarks: BenchmarksConfig = Field(default_factory=BenchmarksConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    inv_grpo: InvGRPOConfig = Field(default_factory=InvGRPOConfig)

    def setup_environment(self) -> None:
        """Apply environment variables to ensure HF & PyTorch respect user config."""
        hf_dir = os.path.abspath(self.storage.hf_cache_dir)
        os.environ["HF_HOME"] = hf_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_dir
        os.environ["HF_DATASETS_CACHE"] = hf_dir
        if os.getenv("HF_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
        self.storage.ensure_dirs(ROOT_DIR)


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load configuration from config.yaml, overlay environment variables, and return validated Settings."""
    cfg_file = Path(config_path) if config_path else ROOT_DIR / "config.yaml"
    raw_data: Dict[str, Any] = {}

    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    raw_data = content
        except Exception as e:
            print(f"[CONFIG WARNING] Failed to parse '{cfg_file}': {e}. Using defaults.")

    # Apply environment variable overrides if present
    storage_dict = raw_data.get("storage", {})
    if os.getenv("HF_HOME") or os.getenv("HF_CACHE_DIR"):
        storage_dict["hf_cache_dir"] = os.getenv("HF_CACHE_DIR") or os.getenv("HF_HOME")
    if os.getenv("LADDER_CACHE_DIR"):
        storage_dict["ladder_cache_dir"] = os.getenv("LADDER_CACHE_DIR")
    if os.getenv("DISTILLATION_CACHE_DIR"):
        storage_dict["distillation_cache_dir"] = os.getenv("DISTILLATION_CACHE_DIR")
    if os.getenv("RESULTS_DIR"):
        storage_dict["results_dir"] = os.getenv("RESULTS_DIR")
    if os.getenv("CHECKPOINTS_DIR"):
        storage_dict["checkpoints_dir"] = os.getenv("CHECKPOINTS_DIR")
    raw_data["storage"] = storage_dict

    models_dict = raw_data.get("models", {})
    if os.getenv("STUDENT_MODEL"):
        models_dict["student_model"] = os.getenv("STUDENT_MODEL")
    if os.getenv("QUANTIZATION"):
        models_dict["quantization"] = os.getenv("QUANTIZATION")
    raw_data["models"] = models_dict

    eval_dict = raw_data.get("evaluation", {})
    if os.getenv("TIMEOUT_SECONDS"):
        try:
            eval_dict["timeout_seconds"] = float(os.getenv("TIMEOUT_SECONDS"))
        except ValueError:
            pass
    raw_data["evaluation"] = eval_dict

    settings = Settings(**raw_data)
    settings.setup_environment()
    return settings


# Global singleton
_SETTINGS: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """Retrieve or initialize the global Settings instance."""
    global _SETTINGS
    if _SETTINGS is None or config_path is not None:
        _SETTINGS = load_settings(config_path)
    return _SETTINGS


# Convenient module-level settings instance
settings = get_settings()

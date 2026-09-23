"""Model inference runner using 4-bit NF4 BitsAndBytes quantization."""

import os
import re
from typing import List, Optional
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.core.interfaces import IModelRunner
from src.core.exceptions import ModelInferenceError, VRAMExceededError
from src.core.config import settings


class QuantizedModelRunner(IModelRunner):
    """Loads SLMs in 4-bit NF4 precision and runs token generation."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device_map: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path or settings.models.student_model
        self.adapter_path = adapter_path
        self.device_map = device_map or settings.models.device_map
        self.cache_dir = cache_dir or settings.storage.hf_cache_dir
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"[MODEL] Loading tokenizer for '{self.model_name_or_path}' (cache: {self.cache_dir})...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True
            )

            print(f"[MODEL] Loading 4-bit NF4 quantized model: '{self.model_name_or_path}'...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )

            if self.adapter_path:
                from peft import PeftModel
                print(f"[MODEL] Loading LoRA adapter weights from '{self.adapter_path}'...")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

            self.model.eval()
            print("[MODEL] Model successfully loaded and ready for inference.")

        except torch.cuda.OutOfMemoryError as e:
            raise VRAMExceededError(f"VRAM exceeded during model loading: {e}") from e
        except Exception as e:
            raise ModelInferenceError(f"Failed to load model '{self.model_name_or_path}': {e}") from e

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        num_samples: int = 1,
        max_new_tokens: int = 1024
    ) -> List[str]:
        """Generate code completions for a single prompt."""
        return self.generate_batch(
            prompts=[prompt],
            temperature=temperature,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
        )[0]

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        num_samples: int = 1,
        max_new_tokens: int = 1024
    ) -> List[List[str]]:
        """Generate code completions for a batch of prompts.

        Pads prompts on the left so they share a single forward pass through
        the model, which is significantly faster than calling generate() in a
        loop.  Returns a list of length len(prompts) where each element is a
        list of ``num_samples`` completions.
        """
        try:
            do_sample = (temperature > 0.0 and num_samples > 1)

            # Apply chat template to every prompt
            formatted: List[str] = []
            for p in prompts:
                if self.adapter_path and ("vanilla" in self.adapter_path.lower() or "sft" in self.adapter_path.lower()):
                    # Vanilla SFT model: use the same ### Problem / ### Solution
                    # format that was used during SFT training so the prompt
                    # distribution matches what the adapter learned.
                    formatted.append(
                        f"### Problem:\n{p}\n\n### Solution:\n"
                    )
                elif hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    messages = [{"role": "user", "content": p}]
                    formatted.append(
                        self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )
                else:
                    formatted.append(p)

            # Left-pad so all sequences have the same length for batched generation
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)
            self.tokenizer.padding_side = original_padding_side

            # Each prompt's generated tokens start after its own input length.
            # With left-padding all inputs share the same padded length, so we
            # can read input_len once from the batch dimension.
            padded_input_len = inputs["input_ids"].shape[1]

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 0.01)
                gen_kwargs["top_p"] = 0.95
                gen_kwargs["num_return_sequences"] = num_samples
            # NOTE: do NOT pass temperature/top_p/top_k when do_sample=False —
            # they are ignored and cause UserWarning noise.

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # outputs shape: (batch_size * num_samples, padded_input_len + new_tokens)
            results: List[List[str]] = []
            for i in range(len(prompts)):
                sample_completions: List[str] = []
                for s in range(num_samples):
                    seq = outputs[i * num_samples + s]
                    generated_tokens = seq[padded_input_len:]
                    decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    sample_completions.append(self._extract_code(decoded))
                results.append(sample_completions)

            return results

        except torch.cuda.OutOfMemoryError as e:
            raise VRAMExceededError(f"VRAM exceeded during batched generation: {e}") from e
        except Exception as e:
            raise ModelInferenceError(f"Batched generation failed: {e}") from e

    def _extract_code(self, raw_text: str) -> str:
        """Extract Python code block from markdown or raw model output.

        Handles three output formats:
        1. ```python ... ``` fenced blocks (preferred)
        2. ``` ... ``` generic fenced blocks
        3. Raw text that may contain SFT training artefacts like
           '### Solution:' headers that the model echoes back.
        """
        # 1. Fenced python block
        code_block_match = re.search(r"```python\s*(.*?)\s*```", raw_text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # 2. Generic fenced block
        generic_block_match = re.search(r"```\s*(.*?)\s*```", raw_text, re.DOTALL)
        if generic_block_match:
            return generic_block_match.group(1).strip()

        # 3. Strip SFT prompt artefacts: the adapter was trained with a
        #    "### Problem: ... ### Solution:" format.  The model sometimes
        #    echoes the header in the completion, which causes a SyntaxError
        #    inside the sandbox.  Remove everything up to and including the
        #    last "### Solution:" marker if present.
        solution_marker = re.search(r"###\s*Solution\s*:\s*", raw_text, re.IGNORECASE)
        if solution_marker:
            raw_text = raw_text[solution_marker.end():]

        return raw_text.strip()

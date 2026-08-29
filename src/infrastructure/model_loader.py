"""Model inference runner using 4-bit NF4 BitsAndBytes quantization."""

import re
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.core.interfaces import IModelRunner
from src.core.exceptions import ModelInferenceError, VRAMExceededError


class QuantizedModelRunner(IModelRunner):
    """Loads SLMs in 4-bit NF4 precision and runs token generation."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
    ):
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"[MODEL] Loading tokenizer from '{self.model_name_or_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
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
                trust_remote_code=True
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
        try:
            # Format as chat prompt if model supports chat template
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                formatted_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_input = prompt

            inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]

            do_sample = (temperature > 0.0 and num_samples > 1)
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
            else:
                gen_kwargs["num_return_sequences"] = 1

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            completions = []
            for output in outputs:
                generated_tokens = output[input_len:]
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                clean_code = self._extract_code(decoded)
                completions.append(clean_code)

            return completions

        except torch.cuda.OutOfMemoryError as e:
            raise VRAMExceededError(f"VRAM exceeded during generation: {e}") from e
        except Exception as e:
            raise ModelInferenceError(f"Generation failed: {e}") from e

    def _extract_code(self, raw_text: str) -> str:
        """Extract Python code block from markdown or raw model output."""
        code_block_match = re.search(r"```python\s*(.*?)\s*```", raw_text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        generic_block_match = re.search(r"```\s*(.*?)\s*```", raw_text, re.DOTALL)
        if generic_block_match:
            return generic_block_match.group(1).strip()

        return raw_text.strip()

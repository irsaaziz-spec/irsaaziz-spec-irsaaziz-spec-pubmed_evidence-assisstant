"""LLM backends. `none` = retrieval-only; `local` = an open model on this machine via transformers.

The local backend reads the model's probabilities for yes/no/maybe after "Decision:" in ONE forward pass
(no text generation), which is what the calibrator was fitted on.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

from .labels import CLASSES


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


class LLMBackend(Protocol):
    name: str
    available: bool

    def score_decision(self, system: str, user: str) -> tuple[list[float], int]: ...
    def generate(self, system: str, user: str, max_new_tokens: int = 100) -> tuple[str, int, int]: ...


class NullBackend:
    """No LLM: the service still returns the retrieved abstracts."""

    name = "none"
    available = False

    def score_decision(self, system: str, user: str) -> tuple[list[float], int]:
        raise RuntimeError("no LLM backend configured")

    def generate(self, system: str, user: str, max_new_tokens: int = 100) -> tuple[str, int, int]:
        raise RuntimeError("no LLM backend configured")


class LocalHFBackend:
    name = "local"
    available = True

    def __init__(self, llm_id: str = "Qwen/Qwen3-4B-Instruct-2507", load_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        self.model = self._load(AutoModelForCausalLM, llm_id, load_4bit)
        self.class_ids = [self._class_token_ids(c) for c in CLASSES]
        flat = [i for ids in self.class_ids for i in ids]
        if len(flat) != len(set(flat)):
            raise RuntimeError(f"answer tokens overlap for this tokenizer: {self.class_ids}")
        self._lock = threading.Lock()  # one GPU, one request at a time

    def _load(self, auto_cls, llm_id: str, load_4bit: bool):
        torch = self._torch
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs: dict = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        last: Exception | None = None
        for dtype_arg in ("dtype", "torch_dtype"):  # argument renamed across transformers versions
            try:
                return auto_cls.from_pretrained(llm_id, **{dtype_arg: dtype}, **kwargs).eval()
            except TypeError as exc:
                last = exc
        raise RuntimeError("could not load the model") from last

    def _class_token_ids(self, word: str) -> list[int]:
        ids = set()
        for variant in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            enc = self.tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        return sorted(ids)

    def _prompt(self, system: str, user: str, suffix: str = "") -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True) + suffix

    def score_decision(self, system: str, user: str) -> tuple[list[float], int]:
        with self._lock, self._torch.inference_mode():
            enc = self.tokenizer(self._prompt(system, user, "Decision:"), return_tensors="pt",
                                 add_special_tokens=False).to(self.model.device)
            try:
                out = self.model(**enc, logits_to_keep=1)
            except TypeError:  # older transformers
                out = self.model(**enc)
            logits = out.logits[0, -1].float().cpu().numpy().astype("float64")
            n_tokens = int(enc["input_ids"].shape[1])
        logp = logits - _logsumexp(logits)
        return [float(_logsumexp(logp[ids])) for ids in self.class_ids], n_tokens

    def generate(self, system: str, user: str, max_new_tokens: int = 100) -> tuple[str, int, int]:
        with self._lock, self._torch.inference_mode():
            enc = self.tokenizer(self._prompt(system, user), return_tensors="pt",
                                 add_special_tokens=False).to(self.model.device)
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
            n_in = int(enc["input_ids"].shape[1])
            new = out[0][n_in:]
            text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        return text, n_in, int(new.shape[0])

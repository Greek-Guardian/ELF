"""PPL / BLEU / ROUGE evaluation metrics (PyTorch).

Translated from src/utils/metrics_utils.py.
No JAX dependencies — only transformers + sacrebleu + rouge_score.
"""

import logging
from typing import Dict, List, Optional

from utils.logging_utils import log_for_0


class Metrics:
    """Generative perplexity evaluator using a frozen GPT-2 (or similar) model."""

    def __init__(
        self,
        gen_ppl_eval_model_name_or_path: str = "gpt2-large",
        eval_ppl_batch_size: int = 64,
        eval_context_size: int = 1024,
    ):
        self.model_name = gen_ppl_eval_model_name_or_path
        self.batch_size = eval_ppl_batch_size
        self.context_size = eval_context_size
        self._model = None
        self._tokenizer = None
        self._texts: List[str] = []

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log_for_0(f"Loading PPL eval model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(device)
            log_for_0(f"PPL eval model loaded on {device}")

    def reset(self):
        self._texts = []

    def record_generative_perplexity(
        self,
        text_samples: List[str],
        max_length: int = 1024,
        retokenize: bool = True,
    ) -> Dict[str, float]:
        """Compute PPL and entropy over `text_samples`."""
        import math
        import torch

        self._load_model()
        device = next(self._model.parameters()).device
        tok = self._tokenizer

        total_nll = 0.0
        total_tokens = 0
        total_entropy = 0.0

        for i in range(0, len(text_samples), self.batch_size):
            batch_texts = text_samples[i : i + self.batch_size]
            enc = tok(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
            # outputs.loss is mean NLL over non-masked tokens
            n_tokens = attention_mask.sum().item()
            total_nll += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

            # Per-token entropy from logits
            import torch.nn.functional as F
            logits = outputs.logits.float()  # (B, S, V)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(-1)  # (B, S)
            mask = attention_mask.bool()
            total_entropy += entropy[mask].sum().item()

        mean_nll = total_nll / max(total_tokens, 1)
        mean_entropy = total_entropy / max(total_tokens, 1)
        return {
            "ppl": math.exp(mean_nll),
            "mean_entropy": mean_entropy,
        }


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Corpus BLEU-4 using sacrebleu."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, [references])
    return result.score


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(
    hypotheses: List[str], references: List[str]
) -> Dict[str, float]:
    """Corpus ROUGE-1/2/L using rouge_score."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = 0
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        for k in totals:
            totals[k] += scores[k].fmeasure
        n += 1
    return {k: v / max(n, 1) * 100 for k, v in totals.items()}

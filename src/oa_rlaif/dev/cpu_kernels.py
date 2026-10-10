"""Smoke test only: a plain-PyTorch stand-in for TRL's Triton log-prob kernel, so SFTTrainer runs on a Mac CPU.

TRL 1.15's SFTTrainer always computes the loss through `trl.kernels.ChunkedLogProbFunction`, which needs Triton
(PyTorch ships it on Linux only). On Lambda the real kernel is used (triton is in requirements-train.lock); this
module is installed only when OA_SMOKE=1 and the kernel is missing. It builds the full logits, which is fine for
the 26M-parameter smoke model and gives the same outputs: (log_probs, entropy, log_sum_sq_probs, mean_logits, is_top1).
"""

from __future__ import annotations

OPTIONAL_OUTPUTS = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")


class TorchChunkedLogProb:
    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None,
              logit_scale=1.0, outputs=OPTIONAL_OUTPUTS):
        import torch.nn.functional as F
        import torch

        z = F.linear(hidden, weight, bias).float() * logit_scale
        if final_logit_softcapping is not None:
            z = final_logit_softcapping * torch.tanh(z / final_logit_softcapping)
        z = z / temperature
        log_p = z.log_softmax(-1)
        log_probs = log_p.gather(-1, targets[:, None]).squeeze(-1)
        entropy = -(log_p.exp() * log_p).sum(-1) if "entropy" in outputs else None
        log_sum_sq = torch.logsumexp(2 * log_p, -1) if "log_sum_sq_probs" in outputs else None
        mean_logits = z.mean(-1) if "mean_logits" in outputs else None
        is_top1 = (z.argmax(-1) == targets) if "is_top1" in outputs else None
        return log_probs, entropy, log_sum_sq, mean_logits, is_top1


def install() -> bool:
    """Patch TRL's missing kernel with the stand-in. Returns True when it patched something."""
    import trl.trainer.utils as utils

    if getattr(utils, "_ChunkedLogProbFunction", None) is None:
        utils._ChunkedLogProbFunction = TorchChunkedLogProb
        return True
    return False

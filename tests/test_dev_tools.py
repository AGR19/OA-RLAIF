"""Modelfile stop token (Gemma 4) and the smoke-test stand-in for TRL's Triton log-prob kernel."""

from pathlib import Path

import pytest


def test_modelfile_uses_gemma4_turn_token(tmp_path: Path):
    from oa_rlaif.export.release import write_modelfile

    text = write_modelfile(tmp_path, "oa-rlaif-e4b-Q4_K_M.gguf").read_text()
    assert "FROM ./oa-rlaif-e4b-Q4_K_M.gguf" in text
    assert 'PARAMETER stop "<turn|>"' in text  # Gemma 4 end of turn (token 106), not Gemma 3's <end_of_turn>
    assert "<end_of_turn>" not in text


def test_cpu_logprob_stand_in_matches_log_softmax():
    torch = pytest.importorskip("torch")
    from oa_rlaif.dev.cpu_kernels import TorchChunkedLogProb

    torch.manual_seed(0)
    hidden, weight = torch.randn(6, 8, requires_grad=True), torch.randn(13, 8)
    targets = torch.randint(0, 13, (6,))
    log_probs, entropy, _, _, is_top1 = TorchChunkedLogProb.apply(hidden, weight, None, targets, 1.0, 4096, 30.0)
    logits = 30.0 * torch.tanh((hidden @ weight.T) / 30.0)
    reference = logits.log_softmax(-1)[torch.arange(6), targets]
    assert torch.allclose(log_probs, reference, atol=1e-6)
    assert torch.all(entropy > 0)
    assert torch.equal(is_top1, logits.argmax(-1) == targets)
    (-log_probs.sum()).backward()
    assert hidden.grad is not None

import asyncio
import time

from oa_rlaif.api_client import RateLimiter, answer_text
from oa_rlaif.reward.judge import Judge


def test_answer_text_strips_inline_thought():
    assert answer_text("<thought>weigh C1 vs C2</thought>\nSCORE: 7") == "SCORE: 7"
    assert answer_text("SCORE: 4") == "SCORE: 4"


def test_answer_text_falls_back_to_truncated_thought():
    assert answer_text("<thought>... so SCORE: 6") == "... so SCORE: 6"


def test_rate_limiter_spaces_requests():
    async def run():
        limiter = RateLimiter(rpm=600)  # 0.1 s apart
        start = time.monotonic()
        for _ in range(4):
            await limiter.wait()
        return time.monotonic() - start
    assert asyncio.run(run()) >= 0.29


def test_judge_fraction_is_deterministic_per_episode(tmp_path):
    judge = Judge(api_key="x", cache_path=tmp_path / "c.sqlite", fraction=0.3)
    picks = [judge.selected({"id": f"ep{i}"}) for i in range(2000)]
    assert picks == [judge.selected({"id": f"ep{i}"}) for i in range(2000)]
    assert 0.25 < sum(picks) / len(picks) < 0.35
    assert asyncio.run(judge.score({"id": next(f"ep{i}" for i, p in enumerate(picks) if not p)}, "x")) == 5.0

# Vendored third-party code

| Project | Version | Commit | Licence | Why vendored |
|---|---|---|---|---|
| [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | v2.4.6 | a83fcae82d2a08f0ee0c688f9d137b3566c097f8 | MIT (`mini-swe-agent/LICENSE.md`) | The agent harness runs from local source on the training box; no PyPI download needed at run time. |
| [JevAgentHarness](https://github.com/JevAgentHarness/JevAgentHarness) | 0.1.0 | f2932199e1ef1d9f089b51d39430c87be0f560b5 (2026-10-16) | MIT (`JevAgentHarness/LICENSE`) | The routing proxy (`jah`) between mini-SWE-agent and the All-merged Gemma tiers, with pplx-decider as its decision model (`dm_harness/`, branch `dm_harness`); installed editable on the GPU box. `research/` is the paper's code, kept for reference; the paper's PDF (`paper/`, arXiv:2610.21412) is not tracked, since the licensed-data gate allows no PDF in git. |

Unmodified copies. Installed editable via `requirements-train.txt` (`-e ./third_party/mini-swe-agent`).
Files such as `AGENTS.md` / `CLAUDE.md` inside vendored projects belong to those projects.

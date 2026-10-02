"""S²-Guidance block drop — a fresh random set of skipped blocks for every model evaluation.

Paper: "S²-Guidance: Stochastic Self Guidance for Training-Free Enhancement of Diffusion Models",
arXiv 2508.12880 (ICLR 2026), main-text form of v3/v4 (Eq. 4, Algorithm 1)::

    D̃ = D(x_t|∅) + λ·(D(x_t|c) − D(x_t|∅)) − ω·(D̂(x_t|c, m_t) − D(x_t|c))
      = CFG + ω·(D_cond − D̂)

``D̂`` is the *conditional* prediction with a random subset ``m_t`` of transformer blocks dropped
(block output = block input — the extension's SLG weak row), drawn anew inside the timestep loop.
The coefficients sum to 1, so the term is the same in x0, ε or v space. (v1/v2 and the v4
appendix use ``… − ω·D̂`` without ``+ω·D_cond``; that form rescales the prediction by ``1 − ω`` and
is treated as an erratum.) No official code was released (AMAP-ML/S2-Guidance has no code and no
LICENSE), so this is a reimplementation from the paper.

Paper settings this module follows:

* drop a fixed number of blocks drawn uniformly without replacement (Table 4 is indexed by the
  count; "≈10%" in the text, while 1–2 of 24 blocks scored best) — ``DEFAULT_RATIO`` 0.05 gives
  1 / 2 / 3 blocks on Anima's 28 / 40 / 52-block models;
* never drop block 0 (Fig. 7c,d: dropping the first block degrades results) — the default
  eligible set is blocks 1..N-1;
* ω = 0.25 (App. C.4, SD3/SD3.5/Wan) — ``DEFAULT_SCALE``;
* a central window of the denoising process (Sec. 4.5, Fig. 7e: "central 80%") — 0.10–0.90.

The draw is reproducible: ``random.Random`` seeded with a string of the generation seed, the pass
("base"/"hires") and the evaluation's draw index (Python seeds a ``str`` through SHA-512, so the
sequence is the same on every machine). One mask is shared by the batch (the paper does not say).
"""

from __future__ import annotations

import random

__all__ = [
    "DEFAULT_END",
    "DEFAULT_RATIO",
    "DEFAULT_SCALE",
    "DEFAULT_START",
    "count_for",
    "default_eligible",
    "draw_blocks",
    "draw_key",
]

DEFAULT_RATIO = 0.05
DEFAULT_SCALE = 0.25
DEFAULT_START = 0.10
DEFAULT_END = 0.90


def default_eligible(blocks: int) -> set[int]:
    """Blocks 1..N-1 — every block but the first (paper Fig. 7c,d)."""
    return set(range(1, int(blocks)))


def count_for(ratio: float, eligible: int) -> int:
    """How many blocks one evaluation drops: ``round(ratio·n)``, at least 1 while ``ratio > 0``."""
    eligible = int(eligible)
    if eligible <= 0 or not ratio or ratio <= 0.0:
        return 0
    return max(1, min(eligible, int(round(float(ratio) * eligible))))


def draw_key(seed, pass_tag: str, draw: int) -> str:
    return f"s2:{int(seed)}:{pass_tag}:{int(draw)}"


def draw_blocks(eligible, ratio: float, seed, pass_tag: str, draw: int) -> set[int]:
    """The blocks dropped at one evaluation — a reproducible random subset of ``eligible``."""
    pool = sorted({int(index) for index in eligible})
    count = count_for(ratio, len(pool))
    if count <= 0:
        return set()
    rng = random.Random(draw_key(seed, pass_tag, draw))
    return set(rng.sample(pool, count))

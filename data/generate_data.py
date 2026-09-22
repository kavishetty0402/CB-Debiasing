"""
Build the demonstration dataset.

This produces a fixed, seeded set of Fed communications (policy statements,
meeting minutes, and official speeches) from 2014 onward. A large corpus is the
natural setting for the method: you can only afford to hand label a small sample,
and the debiasing borrows strength from the thousands of unlabelled items.

Each row has:

  date           communication date
  statement_id   simple identifier
  true_hawk      the ground truth hawkishness on a minus 1 to plus 1 scale.
                 In a real deployment this is what a human produces by reading
                 the statement against a written rubric. The app only ever uses
                 this column for the small hand labelled sample, never for the
                 rest of the corpus.
  ai_promptA     LLM hawkishness score under prompt wording A
  ai_promptB     LLM hawkishness score under prompt wording B
  ai_promptC     LLM hawkishness score under prompt wording C
  yield_change   same day change in the 2 year yield in basis points

The three AI columns imitate the same model asked with three reasonable but
different prompts. Each distorts the truth in its own way (a different slope, a
different baseline shift, and its own noise), which is exactly the sensitivity
Dell and Rambachan warn about: swap the prompt and the naive answer moves.

The true relationship between hawkishness and the yield change is fixed at
BETA_TRUE basis points per unit of hawkishness. A correct method should recover
that number no matter which prompt produced the scores. A naive regression on
the raw AI scores will not.

Everything is reproducible from the seed. Re run this file to rebuild the CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260922
N_STATEMENTS = 2000
BETA_TRUE = 5.0          # basis points of 2y yield move per unit of hawkishness
MARKET_NOISE_SD = 3.0    # basis points of same day noise unrelated to tone

# prompt distortions: (slope on truth, baseline shift, noise sd)
PROMPTS = {
    "ai_promptA": (0.85, 0.20, 0.20),
    "ai_promptB": (1.15, -0.15, 0.20),
    "ai_promptC": (0.95, 0.05, 0.20),
}


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # communication dates from 2014, a couple of business days apart
    dates = pd.bdate_range("2014-01-02", periods=N_STATEMENTS, freq="2B")

    # true hawkishness with mild persistence, then clipped to the scale
    raw = np.zeros(N_STATEMENTS)
    raw[0] = rng.normal(0.1, 0.5)
    for t in range(1, N_STATEMENTS):
        raw[t] = 0.5 * raw[t - 1] + rng.normal(0.05, 0.45)
    true_hawk = np.clip(raw, -1.0, 1.0)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "statement_id": [f"DOC_{i:04d}" for i in range(N_STATEMENTS)],
        "true_hawk": np.round(true_hawk, 4),
    })

    for name, (slope, shift, noise) in PROMPTS.items():
        score = slope * true_hawk + shift + rng.normal(0.0, noise, N_STATEMENTS)
        df[name] = np.round(np.clip(score, -1.5, 1.5), 4)

    yc = BETA_TRUE * true_hawk + rng.normal(0.0, MARKET_NOISE_SD, N_STATEMENTS)
    df["yield_change"] = np.round(yc, 2)

    return df


if __name__ == "__main__":
    out = build()
    out.to_csv("fomc_demo.csv", index=False)
    print(f"Wrote fomc_demo.csv with {len(out)} rows")
    print(out.head())

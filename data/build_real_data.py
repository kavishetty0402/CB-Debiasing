"""
Build the real FOMC dataset (run this once, on your own machine).

This is the offline, heavy step. It never runs inside the Streamlit app. It
produces data/fomc_real.csv, which the app then reads with no model loading at
all, so the deployed app stays light.

What it does:
  1. Loads the Trillion Dollar Words dataset of human labelled FOMC sentences
     (gtfintechlab/fomc_communication). Each sentence has a human hawkish,
     dovish, or neutral label. These human labels are the ground truth.
  2. Runs the purpose built classifier (gtfintechlab/FOMC-RoBERTa) over the same
     sentences. Its output is the AI signal we want to debias. This is a fixed
     classifier, not a prompted model, so there is no prompt to choose.
  3. Writes one row per sentence with the AI score and the human truth.

Why the test split by default:
  RoBERTa was fine tuned on the train split, so its predictions there are
  unrealistically good (it has seen those sentences). Using the test split gives
  a realistic picture of the model's error, which is the honest setting for
  showing why debiasing matters. Change SPLIT below if you want.

Install the build requirements first:
    pip install -r requirements-build.txt

Then run from the repo root:
    python data/build_real_data.py

Output:
    data/fomc_real.csv     columns: statement_id, year, sentence, true_hawk, ai_score

After it is written, commit the CSV and push. Real mode in the app will light up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "ProsusAI/finbert"
DATASET_NAME = "gtfintechlab/fomc_communication"
SPLIT = "test"   # "test" (realistic error), "train", or "all"

# Model label indices, from the model card:
#   index 0 = Dovish, index 1 = Hawkish, index 2 = Neutral
# The dataset uses the same integer scheme. If the printed sanity check below
# looks wrong (for example almost everything comes out neutral), flip this map.
LABEL_TO_SCORE = {0: -1.0, 1: 1.0, 2: 0.0}   # dovish, hawkish, neutral

OUT_PATH = Path(__file__).parent / "fomc_real.csv"
BATCH = 32


def load_sentences() -> pd.DataFrame:
    from datasets import load_dataset, concatenate_datasets

    ds = load_dataset(DATASET_NAME)
    if SPLIT == "all":
        parts = [ds[s] for s in ds.keys()]
        data = concatenate_datasets(parts)
    else:
        data = ds[SPLIT]

    df = data.to_pandas()
    # column names in this dataset: sentence, year, label
    df = df.rename(columns={"sentence": "sentence", "year": "year", "label": "label"})
    df = df.dropna(subset=["sentence", "label"]).reset_index(drop=True)
    return df


def score_with_roberta(sentences):
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    )

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, do_lower_case=True, do_basic_tokenize=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    _ = AutoConfig.from_pretrained(MODEL_NAME)
    model.eval()
    if device == 0:
        model = model.cuda()

    p_hawk = np.zeros(len(sentences))
    p_dove = np.zeros(len(sentences))
    ai_label = np.zeros(len(sentences), dtype=int)

    with torch.no_grad():
        for start in range(0, len(sentences), BATCH):
            chunk = list(sentences[start:start + BATCH])
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=256, return_tensors="pt")
            if device == 0:
                enc = {k: v.cuda() for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # FinBERT order: 0 positive, 1 negative, 2 neutral
            # We read positive as the hawkish-leaning side and negative as dovish.
            p_hawk[start:start + len(chunk)] = probs[:, 0]
            p_dove[start:start + len(chunk)] = probs[:, 1]
            ai_label[start:start + len(chunk)] = probs.argmax(axis=1)
            print(f"  scored {min(start + BATCH, len(sentences))} / {len(sentences)}")

    ai_score = p_hawk - p_dove   # positive means hawkish, negative means dovish
    return ai_score, ai_label


def main():
    print(f"Loading dataset {DATASET_NAME} split={SPLIT} ...")
    df = load_sentences()
    print(f"  {len(df)} sentences")

    print(f"Loading model {MODEL_NAME} and scoring (first run downloads weights) ...")
    ai_score, ai_label = score_with_roberta(df["sentence"].tolist())

    true_hawk = df["label"].map(LABEL_TO_SCORE).astype(float)

    out = pd.DataFrame({
        "statement_id": [f"FOMC_S_{i:04d}" for i in range(len(df))],
        "year": df["year"].values,
        "sentence": df["sentence"].str.slice(0, 300).values,
        "true_hawk": np.round(true_hawk.values, 3),
        "ai_score": np.round(ai_score, 4),
    })
    out.to_csv(OUT_PATH, index=False)

    # sanity check so you can eyeball the label mapping and model behaviour
    agree = (np.sign(out["ai_score"]) == np.sign(out["true_hawk"])).mean()
    print()
    print(f"Wrote {OUT_PATH} with {len(out)} rows")
    print(f"  true label mix: {df['label'].value_counts().to_dict()}")
    print(f"  mean true hawkishness: {out['true_hawk'].mean():+.3f}")
    print(f"  mean AI score:         {out['ai_score'].mean():+.3f}")
    print(f"  sign agreement AI vs human: {agree:.2%}")
    print()
    print("If sign agreement is very low, the label map is probably flipped. "
          "Adjust LABEL_TO_SCORE at the top and rerun.")


if __name__ == "__main__":
    main()

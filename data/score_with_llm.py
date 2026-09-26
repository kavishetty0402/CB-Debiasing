"""
Score the real FOMC sentences with a language model, three ways.

Run this once, on your own Mac. It never runs inside the Streamlit app.

What it does:
  1. Reads data/fomc_real.csv (the 496 real FOMC sentences with human labels).
  2. Reloads the full sentence text from the dataset, because the CSV cuts long
     sentences at 300 characters.
  3. Asks a local language model to label every sentence HAWKISH, DOVISH or
     NEUTRAL, using three differently worded questions (prompt A, B, C).
  4. Adds three new columns to data/fomc_real.csv: ai_promptA, ai_promptB,
     ai_promptC. The FinBERT column (ai_score) is kept.

After that, the app's Real FOMC data mode shows the prompt stress test on real
Fed sentences: does rewording the question move the naive answer, and does the
debiased answer stay put?

The model runs locally through Ollama. Free, no account, no API key.

One time setup:
  1. Install Ollama from https://ollama.com and open it.
  2. In Terminal:  ollama pull llama3.1:8b

Then, from the project folder:
  python3 data/score_with_llm.py

It takes a while (roughly 1,500 questions). Progress is saved as it goes, so if
you stop it or it crashes, just run the same command again and it carries on
from where it stopped.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "llama3.1:8b"      # a lighter option for older Macs: "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

HERE = Path(__file__).parent
REAL_PATH = HERE / "fomc_real.csv"
PROGRESS_PATH = HERE / "llm_progress.csv"
SAVE_EVERY = 25

# Three wordings of the same question. Same rubric, different phrasing.
PROMPTS = {}

# A: the wording used in the Trillion Dollar Words zero shot benchmark.
PROMPTS["ai_promptA"] = (
    "Classify the following sentence from the FOMC into HAWKISH, DOVISH, or "
    "NEUTRAL. Label HAWKISH if it corresponds to tightening of monetary policy, "
    "DOVISH if it corresponds to easing of monetary policy, or NEUTRAL if the "
    "stance is neutral. Answer with one word only.\n\n"
    "Sentence: {sentence}"
)

# B: the same question, framed from a bond trader's point of view.
PROMPTS["ai_promptB"] = (
    "You are a US rates trader reading Federal Reserve communication. After "
    "reading this sentence, would you expect the Fed to raise rates (HAWKISH), "
    "cut rates (DOVISH), or neither (NEUTRAL)? Answer with one word only.\n\n"
    "Sentence: {sentence}"
)

# C: a short, bare instruction.
PROMPTS["ai_promptC"] = (
    "Fed sentence: {sentence}\n\n"
    "Is this HAWKISH, DOVISH or NEUTRAL? One word."
)

WORD_TO_SCORE = {"HAWKISH": 1.0, "DOVISH": -1.0, "NEUTRAL": 0.0}


def ask_model(prompt_text):
    """Send one question to the local model and return its reply as text."""
    body = {
        "model": MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8},
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.loads(response.read().decode("utf-8"))
    return reply["response"]


def reply_to_score(reply_text):
    """Turn the model's reply into +1, -1 or 0. Returns None if unreadable."""
    upper = reply_text.upper()
    for word in ["HAWKISH", "DOVISH", "NEUTRAL"]:
        if word in upper:
            return WORD_TO_SCORE[word]
    return None


def load_full_sentences(df):
    """Reload untruncated sentences from the dataset, in the same row order."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gtfintechlab/fomc_communication")["test"].to_pandas()
    except Exception as error:
        print(f"Could not reload the dataset ({error}). Using the CSV sentences.")
        return list(df["sentence"])
    ds = ds.dropna(subset=["sentence", "label"]).reset_index(drop=True)

    if len(ds) != len(df):
        print("Row counts differ, falling back to the sentences in the CSV.")
        return list(df["sentence"])

    full = []
    for i in range(len(df)):
        short = str(df.loc[i, "sentence"])
        long = str(ds.loc[i, "sentence"])
        if long[:50] == short[:50]:
            full.append(long)
        else:
            full.append(short)
    return full


def check_ollama():
    try:
        ask_model("Reply with the word OK.")
    except Exception as error:
        print("Could not reach Ollama. Is the Ollama app open, and did you run")
        print(f"'ollama pull {MODEL}'?  Details: {error}")
        raise SystemExit(1)


def main():
    df = pd.read_csv(REAL_PATH)
    print(f"Loaded {len(df)} sentences from {REAL_PATH.name}")

    print("Reloading full sentence text from the dataset ...")
    sentences = load_full_sentences(df)

    print(f"Checking that Ollama is running with {MODEL} ...")
    check_ollama()

    # load earlier progress if there is any
    if PROGRESS_PATH.exists():
        progress = pd.read_csv(PROGRESS_PATH)
        print(f"Resuming: {len(progress)} answers already saved.")
    else:
        progress = pd.DataFrame(columns=["row", "prompt", "reply", "score"])

    done = set()
    for i in range(len(progress)):
        done.add((int(progress.loc[i, "row"]), progress.loc[i, "prompt"]))

    new_rows = []
    total = len(df) * len(PROMPTS)
    count = len(done)

    for prompt_name in PROMPTS:
        template = PROMPTS[prompt_name]
        for i in range(len(df)):
            if (i, prompt_name) in done:
                continue
            question = template.format(sentence=sentences[i])
            reply = ask_model(question)
            score = reply_to_score(reply)
            new_rows.append({"row": i, "prompt": prompt_name,
                             "reply": reply.strip(), "score": score})
            count = count + 1

            if len(new_rows) >= SAVE_EVERY:
                progress = pd.concat([progress, pd.DataFrame(new_rows)], ignore_index=True)
                progress.to_csv(PROGRESS_PATH, index=False)
                new_rows = []
                print(f"  {count} / {total} answered")

    if len(new_rows) > 0:
        progress = pd.concat([progress, pd.DataFrame(new_rows)], ignore_index=True)
        progress.to_csv(PROGRESS_PATH, index=False)
    print(f"  {count} / {total} answered")

    # build one column per prompt
    print()
    for prompt_name in PROMPTS:
        column = []
        unreadable = 0
        for i in range(len(df)):
            match = progress[(progress["row"] == i) & (progress["prompt"] == prompt_name)]
            value = match["score"].iloc[-1]
            if pd.isna(value):
                unreadable = unreadable + 1
                value = 0.0   # unreadable reply counts as neutral
            column.append(float(value))
        df[prompt_name] = column

        truth = df["true_hawk"].values
        scores = np.array(column)
        exact = (scores == truth).mean()
        correlation = np.corrcoef(scores, truth)[0, 1]
        print(f"{prompt_name}: exact match {exact:.1%}, correlation {correlation:+.3f}, "
              f"average {scores.mean():+.3f} (truth {truth.mean():+.3f}), "
              f"unreadable replies {unreadable}")

    df.to_csv(REAL_PATH, index=False)
    print()
    print(f"Updated {REAL_PATH.name} with columns {list(PROMPTS)}.")
    print("Upload this file to the data folder on GitHub to update the app.")


if __name__ == "__main__":
    main()

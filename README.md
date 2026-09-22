# Debiasing AI Signals

A small Streamlit tool that shows why raw AI scores can mislead you, and how a
small hand labelled sample fixes it. It is a live, interactive walkthrough of the
correction described in Dell and Rambachan (2026), "The Measurement Revolution?
Credible Measurement and Inference in the Age of AI" (NBER Working Paper 35744).

## The idea in one line

An AI scores every document. You hand label a small random sample. That sample
tells you how the AI errs, so you can correct the final estimate. The correction
stays valid even when the AI is badly biased. A more accurate AI only buys you a
tighter confidence interval, not a more correct answer. The truth must come from
a human applying a rubric, never from another AI.

## Three data sources

Pick one in the sidebar.

1. Simulation. The corpus is generated with a known truth, so you can watch the
   method recover it, and stress test across three prompts. This is the proof
   that the method works, which real data alone cannot give you.
2. Real FOMC data. Genuine Fed sentences from the Trillion Dollar Words dataset,
   scored by the FOMC-RoBERTa classifier as the AI signal, with the dataset's
   human hawkish or dovish labels as the truth. Built offline (see below) and
   frozen into a CSV, so the app never runs the model and stays light.
3. Upload. Bring your own CSV with an AI score column, a true label column (blank
   where not labelled), and an optional outcome column.

## What the app shows

Ten steps, from loading a corpus to exporting a one page report. The panels:

- Average hawkishness across the corpus: naive versus labels only versus
  debiased, checked against a known truth where one exists.
- A downstream regression, hawkishness on the outcome, where the AI score is the
  regressor (the harder errors in variables case). Shown when the data has an
  outcome column, which the simulation does.
- A prompt stress test: the naive answer swings as you reword the prompt, the
  debiased answer stays put. Shown when the data has multiple prompt columns,
  which the simulation does.
- An efficiency panel: how the interval narrows as you label more, and how many
  hand labels your current sample is worth.

## Method note

Dell and Rambachan frame inference with AI predictions as a missing data problem
(their MAR-S framework). The mean case coincides with prediction powered
inference (Angelopoulos et al., 2023). For a regression with an AI scored
regressor, the app solves the corrected estimating equation directly and reports
a sandwich confidence interval, which handles the errors in variables case that
the classic prediction powered routine does not.

## Run it locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Rebuild the simulation dataset (optional):

```
cd data
python generate_data.py
```

## Build the real FOMC data (run once, on your machine)

This is the heavy step. It never runs inside the app. It downloads the classifier
and the dataset, scores the sentences, and writes `data/fomc_real.csv`. You need
internet and a few GB of disk for the model weights. A GPU helps but is not
required.

```
pip install -r requirements-build.txt
python data/build_real_data.py
```

The script prints a sanity check (label mix, mean scores, and how often the AI
agrees with the human labels) so you can confirm the label mapping is right. Once
`data/fomc_real.csv` exists, the Real FOMC data mode works. Commit and push the
CSV to make it appear on the deployed app.

Notes:

- The script uses the dataset's test split by default, because the classifier was
  fine tuned on the train split and would look unrealistically accurate there.
- The real panel targets average hawkishness, not a yield regression. The dataset
  is sentence level with only a year, so there is no reliable date to align to
  daily yields. Aligning release dates and pulling the 2 year yield from FRED is a
  clean next step if you want the real regression too.

## Deploy to Streamlit Community Cloud

1. Create a public GitHub repository and put every file in this folder at its
   root, keeping the layout. If you use the website, use Add file, Upload files,
   and drag the contents in. If `.streamlit` does not upload, create the file
   `.streamlit/config.toml` directly on GitHub (typing the slash makes the
   folder) and paste in the config.
2. Go to https://share.streamlit.io, sign in with GitHub, click New app.
3. Choose your repository, branch `main`, main file `streamlit_app.py`, Deploy.

The first build installs `requirements.txt` only, which is light. No secrets or
API keys are needed. The simulation runs out of the box; Real FOMC data appears
once you build and commit the CSV.

## Files

```
streamlit_app.py          the app and the ten step walkthrough
ppi/estimators.py         the debiasing maths (mean and regression) and efficiency
data/generate_data.py     seeded generator for the simulation dataset
data/fomc_demo.csv        the bundled simulation corpus
data/build_real_data.py   offline builder for the real FOMC dataset
requirements.txt          light dependencies for the app
requirements-build.txt    heavy dependencies for the offline builder only
.streamlit/config.toml    accessible high contrast theme
```

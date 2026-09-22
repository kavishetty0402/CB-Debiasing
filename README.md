# Debiasing AI Signals

A small Streamlit tool that shows why raw AI scores can mislead you, and how a
small hand labelled sample fixes it. It is a live, interactive walkthrough of the
correction described in Dell and Rambachan (2026), "The Measurement Revolution?
Credible Measurement and Inference in the Age of AI" (NBER Working Paper 35744).

## The idea in one line

An AI scores every document. You hand label a small random sample. That sample
tells you how the AI errs, so you can correct the final estimate. The correction
stays valid even when the AI is badly biased. A more accurate AI only buys you a
tighter confidence interval, not a more correct answer.

## What the app shows

The app walks through ten steps, from loading a corpus to exporting a one page
report. The headline panels are:

1. Average hawkishness across the corpus: naive versus labels only versus
   debiased, checked against a known truth.
2. The downstream regression, hawkishness on the change in the outcome, where the
   AI score is the regressor (the harder errors in variables case).
3. A prompt stress test: the naive answer swings as you reword the prompt, while
   the debiased answer stays put.
4. An efficiency panel: how the confidence interval narrows as you label more,
   and how many hand labels your current sample is worth.

The built in demonstration corpus is simulated so the effect is easy to see and
the true answer is known. Every statistic is computed live. You can also upload
your own scored data.

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

To rebuild the demonstration dataset from scratch:

```
cd data
python generate_data.py
```

## Deploy to Streamlit Community Cloud

1. Create a new public repository on GitHub, for example `debiasing-ai-signals`.
2. Put every file in this folder at the root of that repository, keeping the
   folder layout intact. From this folder:

   ```
   git init
   git add .
   git commit -m "Debiasing AI signals demo"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/debiasing-ai-signals.git
   git push -u origin main
   ```

3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click New app, choose your repository, set the branch to `main`, and set the
   main file path to `streamlit_app.py`.
5. Click Deploy. The first build installs the requirements and can take a couple
   of minutes. After that you get a public link you can share.

No secrets or API keys are needed. The demo runs entirely on the bundled data.

## Upload your own data

Switch the data source to upload in the sidebar. Your CSV needs three columns:

- an AI score column (the model's number for each item)
- a true label column, filled in only for the rows you hand labelled and left
  blank for the rest
- an outcome column (the market variable you want to relate the score to)

You map those columns in the sidebar after uploading.

## Files

```
streamlit_app.py        the app and the ten step walkthrough
ppi/estimators.py       the debiasing maths (mean and regression) and efficiency
data/generate_data.py   seeded generator for the demonstration dataset
data/fomc_demo.csv       the bundled demonstration corpus
.streamlit/config.toml  accessible high contrast theme
requirements.txt        dependencies
```

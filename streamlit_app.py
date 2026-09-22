"""
Debiasing AI Signals: a live walkthrough.

A small tool that shows why raw AI scores can mislead you, and how a small hand
labelled sample fixes it. Built around Dell and Rambachan (2026), NBER 35744.

Three data sources:
  1. Simulation. The truth is known, so you can watch the method recover it, and
     you can stress test across three prompts. This is the proof.
  2. Real FOMC data. Real sentences scored by the FOMC-RoBERTa classifier, with
     human hawkish or dovish labels as the truth. Built offline by
     data/build_real_data.py and frozen into data/fomc_real.csv, so the app never
     runs the model itself and stays light.
  3. Upload. Bring your own scored data.

Every statistic is computed live.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from ppi import (
    naive_mean, labeled_only_mean, ppi_mean,
    naive_ols, labeled_only_ols, ppi_ols,
    effective_sample_size,
)

HERE = Path(__file__).parent
DEMO_PATH = HERE / "data" / "fomc_demo.csv"
REAL_PATH = HERE / "data" / "fomc_real.csv"

DEMO_PROMPTS = ["ai_promptA", "ai_promptB", "ai_promptC"]
PROMPT_LABELS = {
    "ai_promptA": "Prompt A (reads slightly more hawkish)",
    "ai_promptB": "Prompt B (reads slightly more dovish)",
    "ai_promptC": "Prompt C (fairly balanced)",
    "ai_score": "FOMC-RoBERTa score",
    "ai_uploaded": "Your uploaded AI score",
}
BLUE = "#1f4e79"
RED = "#c0392b"

st.set_page_config(page_title="Debiasing AI Signals", page_icon="\U0001F4CF", layout="wide")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def draw_labeled(n_rows: int, n_labeled: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_rows)
    return np.sort(idx[:n_labeled]), np.sort(idx[n_labeled:])


def ci_chart(rows, value_title: str, truth: float | None = None, unit: str = ""):
    data = pd.DataFrame(rows)
    order = list(data["method"])
    base = alt.Chart(data)
    rule = base.mark_rule(strokeWidth=3, color=BLUE).encode(
        y=alt.Y("method:N", sort=order, title=None),
        x=alt.X("low:Q", title=f"{value_title} {unit}".strip()), x2="high:Q",
    )
    point = base.mark_point(size=140, filled=True, color=BLUE).encode(
        y=alt.Y("method:N", sort=order), x="estimate:Q",
        tooltip=[alt.Tooltip("method:N", title="Method"),
                 alt.Tooltip("estimate:Q", title="Estimate", format=".3f"),
                 alt.Tooltip("low:Q", title="Low", format=".3f"),
                 alt.Tooltip("high:Q", title="High", format=".3f")],
    )
    label = base.mark_text(align="left", dx=10, dy=-12, color=BLUE).encode(
        y=alt.Y("method:N", sort=order), x="estimate:Q",
        text=alt.Text("estimate:Q", format=".2f"),
    )
    layers = [rule, point, label]
    if truth is not None:
        tdf = pd.DataFrame({"truth": [truth]})
        layers.append(alt.Chart(tdf).mark_rule(
            strokeDash=[6, 4], color=RED, strokeWidth=2).encode(x="truth:Q"))
    return alt.layer(*layers).properties(height=180)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Controls")

source = st.sidebar.radio(
    "Data source",
    ["Simulation (known truth)", "Real FOMC data", "Upload your own CSV"],
    help="Simulation proves the method recovers a known truth. Real FOMC data "
         "runs it on genuine Fed sentences. Upload brings your own.",
)

conf = st.sidebar.select_slider(
    "Confidence level", options=[0.90, 0.95, 0.99], value=0.95,
    format_func=lambda x: f"{int(x * 100)} percent",
)
alpha = 1.0 - conf

seed = st.sidebar.number_input(
    "Random seed for the labelled draw", min_value=0, max_value=9999, value=7, step=1,
    help="Changing this draws a different random validation sample.",
)


# ---------------------------------------------------------------------------
# Resolve the data source into a common shape
# ---------------------------------------------------------------------------

mode = ("sim" if source.startswith("Simulation")
        else "real" if source.startswith("Real") else "upload")

true_col = "true_hawk"
outcome = None
true_mean_ref = None
beta_ref = None

if mode == "sim":
    df = load_csv(str(DEMO_PATH)).copy()
    score_cols = DEMO_PROMPTS
    truth_all = True
    outcome = df["yield_change"].to_numpy(float)
    true_mean_ref = float(df[true_col].mean())
    beta_ref = 5.0

elif mode == "real":
    if not REAL_PATH.exists():
        st.title("Debiasing AI Signals")
        st.header("Real FOMC data has not been built yet")
        st.markdown(
            "This mode reads `data/fomc_real.csv`, produced offline by the builder "
            "script. It is not in the repo yet. Build it once on a machine with "
            "internet, commit the CSV, and this mode will light up. The deployed "
            "app never runs the model, so it stays light."
        )
        st.code(
            "pip install -r requirements-build.txt\n"
            "python data/build_real_data.py\n"
            "git add data/fomc_real.csv && git commit -m 'Add real FOMC data' && git push",
            language="bash",
        )
        st.caption(
            "The builder runs FOMC-RoBERTa (gtfintechlab/FOMC-RoBERTa) over the "
            "Trillion Dollar Words dataset and freezes the scores plus the human "
            "labels. Meanwhile you can use Simulation or Upload in the sidebar."
        )
        st.stop()
    df = load_csv(str(REAL_PATH)).copy()
    score_cols = ["ai_score"]
    truth_all = True
    true_mean_ref = float(df[true_col].mean())

else:  # upload
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        st.title("Debiasing AI Signals")
        st.info(
            "Upload a CSV in the sidebar, or switch to Simulation or Real FOMC "
            "data. Your file needs an AI score column, a true label column (blank "
            "where not labelled), and an outcome column."
        )
        st.stop()
    df = pd.read_csv(uploaded)
    st.sidebar.markdown("Map your columns")
    cols = list(df.columns)
    score_col = st.sidebar.selectbox("AI score column", cols)
    tcol = st.sidebar.selectbox("True label column (blank where unlabelled)", cols)
    ocol = st.sidebar.selectbox("Outcome column (optional)", ["(none)"] + cols)
    df = df.rename(columns={score_col: "ai_uploaded", tcol: "true_hawk"})
    score_cols = ["ai_uploaded"]
    truth_all = False
    if ocol != "(none)":
        outcome = df[ocol].to_numpy(float)

n_rows = len(df)
has_prompts = len(score_cols) > 1
has_outcome = outcome is not None


# ---------------------------------------------------------------------------
# Title and framing
# ---------------------------------------------------------------------------

st.title("Debiasing AI Signals")
st.markdown(
    "A raw AI score is a measurement with unknown error, not a fact. Feed it "
    "straight into an estimate and your answer can be biased and your confidence "
    "interval too narrow, so you end up confidently wrong. The fix from Dell and "
    "Rambachan (2026): hand label a small random sample, learn how the AI errs, "
    "and correct the final estimate. It stays valid even when the AI is poor."
)
if mode == "sim":
    st.caption(
        "Simulation. The corpus is generated with a known truth, so you can check "
        "that the method recovers it and stress test across three prompts. Every "
        "statistic is computed live."
    )
elif mode == "real":
    st.caption(
        "Real FOMC data. Genuine Fed sentences from the Trillion Dollar Words "
        "dataset, scored by the FOMC-RoBERTa classifier (the AI signal), with the "
        "dataset's human hawkish or dovish labels as the truth. We hide most human "
        "labels to simulate hand labelling, then check recovery against the rest."
    )


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------

st.header("Step 1. Load the corpus")
st.write(f"The corpus has {n_rows} items, each with an AI score. "
         + ("The market outcome is attached too." if has_outcome else
            "There is no market outcome in this build, so we focus on the average."))
st.dataframe(df.head(8), width="stretch")


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------

st.header("Step 2. Write the rubric (the one step that needs your judgement)")
st.write(
    "Pin down what you are measuring before you look at any result, so you cannot "
    "quietly tune the definition to fit the answer. This is the anchor the whole "
    "method rests on, and it is why the truth must come from a human applying a "
    "rubric, never from another AI."
)
st.text_area(
    "Hawkishness rubric",
    "Score each item from minus 1 to plus 1.\n"
    "Plus 1: leans toward tighter policy or names inflation as the main risk.\n"
    "Zero: balanced, no clear lean.\n"
    "Minus 1: leans toward easing or names growth and jobs as the main risk.\n"
    "Edge case: naming inflation risk while holding steady counts as mildly hawkish.",
    height=140,
)


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------

st.header("Step 3. The AI scores everything")
if has_prompts:
    st.write(
        "The same model was asked with three different prompts. Pick which to "
        "treat as your signal. This choice should not change your final answer, "
        "but for the naive method it does."
    )
    prompt = st.selectbox("AI scoring to use", score_cols,
                          format_func=lambda c: PROMPT_LABELS.get(c, c), index=2)
else:
    prompt = score_cols[0]
    st.write(f"Signal: {PROMPT_LABELS.get(prompt, prompt)}. A fixed classifier "
             "has no prompt to vary, so there is a single score per item.")

score = df[prompt].to_numpy(float)
hist = alt.Chart(pd.DataFrame({"score": score})).mark_bar(color=BLUE).encode(
    x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="AI hawkishness score"),
    y=alt.Y("count()", title="Number of items"),
).properties(height=200)
st.altair_chart(hist, width="stretch")


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------

st.header("Step 4. Hand label a small random sample")
true_all = df[true_col].to_numpy(float)

if truth_all:
    st.write(
        "Draw a random sample and score it against the rubric. Here the careful "
        "human labels are known, so we reveal them only for the sampled rows. "
        "Nothing outside the sample ever uses the true value."
    )
    max_lab = int(min(n_rows - 5, 500))
    default_lab = int(min(150, max_lab))
    n_labeled = st.slider("Number of items you hand label", 20, max_lab,
                          default_lab, step=10)
    lab_idx, unlab_idx = draw_labeled(n_rows, n_labeled, seed)
else:
    labeled_mask = df[true_col].notna().to_numpy()
    lab_idx = np.where(labeled_mask)[0]
    unlab_idx = np.where(~labeled_mask)[0]
    n_labeled = int(labeled_mask.sum())
    st.write(f"Your file has {n_labeled} rows with a true label and "
             f"{len(unlab_idx)} without. The labelled rows are the validation sample.")

if len(lab_idx) < 5 or len(unlab_idx) < 5:
    st.warning("Need at least a few labelled and a few unlabelled rows to proceed.")
    st.stop()

y_lab = true_all[lab_idx]
f_lab = score[lab_idx]
f_unlab = score[unlab_idx]
st.caption(f"Validation sample: {len(lab_idx)} items. AI scored corpus: {n_rows} items.")


# ---------------------------------------------------------------------------
# Step 5: the mean
# ---------------------------------------------------------------------------

st.header("Step 5. Correct, and see the answers side by side")
st.write(
    "Estimand: the average hawkishness across the corpus. Naive trusts the AI. "
    "Labels only ignores the AI and uses just your sample. Debiased uses both."
)
nm = naive_mean(score, alpha=alpha)
lm = labeled_only_mean(y_lab, alpha=alpha)
pm = ppi_mean(y_lab, f_lab, f_unlab, alpha=alpha)
mean_rows = [
    {"method": "Naive (trust the AI)", **{k: nm[k] for k in ("estimate", "low", "high")}},
    {"method": "Labels only", **{k: lm[k] for k in ("estimate", "low", "high")}},
    {"method": "Debiased", **{k: pm[k] for k in ("estimate", "low", "high")}},
]
st.altair_chart(ci_chart(mean_rows, "Average hawkishness", truth=true_mean_ref),
                width="stretch")
if true_mean_ref is not None:
    st.markdown(
        f"How to read this. The red dashed line is the known truth "
        f"({true_mean_ref:+.3f}). A good method lands on it with an honest "
        f"interval. Naive can miss while looking falsely certain. Labels only is "
        f"honest but wider, because it ignores the corpus. Debiased uses both."
    )


# ---------------------------------------------------------------------------
# Step 6: regression, only when an outcome exists
# ---------------------------------------------------------------------------

no = lo = po = None
if has_outcome:
    o_lab = outcome[lab_idx]
    o_unlab = outcome[unlab_idx]
    st.header("Step 6. Ask the downstream question")
    st.write(
        "Does a more hawkish item move the outcome? We regress the outcome on "
        "hawkishness. The AI score is the regressor, the harder errors in "
        "variables case, and the debiasing corrects for it."
    )
    no = naive_ols(outcome, score, alpha=alpha)
    lo = labeled_only_ols(o_lab, y_lab, alpha=alpha)
    po = ppi_ols(o_lab, y_lab, f_lab, o_unlab, f_unlab, alpha=alpha)
    slope_rows = [
        {"method": "Naive (trust the AI)", **{k: no[k] for k in ("estimate", "low", "high")}},
        {"method": "Labels only", **{k: lo[k] for k in ("estimate", "low", "high")}},
        {"method": "Debiased", **{k: po[k] for k in ("estimate", "low", "high")}},
    ]
    st.altair_chart(ci_chart(slope_rows, "Effect on the outcome per unit of hawkishness",
                             truth=beta_ref), width="stretch")
    if beta_ref is not None:
        st.markdown(
            f"How to read this. The truth is {beta_ref:.1f}. The naive slope is "
            f"pulled toward zero because the AI score is noisy, so it understates "
            f"the response. The debiased slope recovers the true effect."
        )


# ---------------------------------------------------------------------------
# Step 8: prompt stress test, only when there are multiple prompts
# ---------------------------------------------------------------------------

if has_prompts:
    st.header("Step 8. Stress test the prompt")
    st.write(
        "Run the same rubric under all three prompts. The naive answer swings with "
        "the wording while the debiased answer stays put. That stability is the "
        "point: your conclusion should not depend on how you phrased the prompt."
    )
    rows = []
    for c in score_cols:
        sc = df[c].to_numpy(float)
        nm_c = naive_mean(sc, alpha=alpha)
        pm_c = ppi_mean(y_lab, sc[lab_idx], sc[unlab_idx], alpha=alpha)
        tag = c.replace("ai_prompt", "Prompt ")
        rows.append({"prompt": tag, "kind": "Naive", **{k: nm_c[k] for k in ("estimate", "low", "high")}})
        rows.append({"prompt": tag, "kind": "Debiased", **{k: pm_c[k] for k in ("estimate", "low", "high")}})
    sdf = pd.DataFrame(rows)
    color = alt.Color("kind:N", title="Method",
                      scale=alt.Scale(domain=["Naive", "Debiased"], range=[RED, BLUE]))
    err = alt.Chart(sdf).mark_rule(strokeWidth=3).encode(
        x=alt.X("prompt:N", title=None), y=alt.Y("low:Q", title="Average hawkishness"),
        y2="high:Q", color=color, xOffset="kind:N")
    pts = alt.Chart(sdf).mark_point(size=120, filled=True).encode(
        x="prompt:N", y="estimate:Q", color=color, xOffset="kind:N",
        tooltip=["prompt", "kind", alt.Tooltip("estimate:Q", format=".3f")])
    tline = alt.Chart(pd.DataFrame({"t": [true_mean_ref]})).mark_rule(
        strokeDash=[6, 4], color="#333").encode(y="t:Q")
    st.altair_chart((err + pts + tline).properties(height=320), width="stretch")
    nr = sdf[sdf.kind == "Naive"]["estimate"]
    dr = sdf[sdf.kind == "Debiased"]["estimate"]
    st.markdown(
        f"The naive estimate ranges over {nr.max() - nr.min():.3f} across the "
        f"prompts. The debiased estimate ranges over only {dr.max() - dr.min():.3f}, "
        f"clustered on the truth."
    )


# ---------------------------------------------------------------------------
# Step 9: efficiency
# ---------------------------------------------------------------------------

st.header("Step 9. Is it worth it")
st.write(
    "Debiasing borrows strength from the unlabelled corpus, so a few labels can "
    "carry the weight of many more. This shows how the interval narrows as you "
    "label more, and what your current sample is worth."
)
neff_mean = effective_sample_size(n_labeled, lm["se"] ** 2, pm["se"] ** 2)
c1, c2 = st.columns(2)
c1.metric("Labels you actually did", f"{n_labeled}")
c2.metric("Debiased mean is worth about", f"{neff_mean:.0f} hand labels")

if truth_all:
    grid = list(range(20, int(min(n_rows - 5, 500)) + 1, 20))
    curve = []
    for g in grid:
        li, ui = draw_labeled(n_rows, g, seed)
        pm_g = ppi_mean(true_all[li], score[li], score[ui], alpha=alpha)
        lm_g = labeled_only_mean(true_all[li], alpha=alpha)
        curve.append({"labels": g, "width": pm_g["high"] - pm_g["low"], "method": "Debiased"})
        curve.append({"labels": g, "width": lm_g["high"] - lm_g["low"], "method": "Labels only"})
    cdf = pd.DataFrame(curve)
    line = alt.Chart(cdf).mark_line(point=True).encode(
        x=alt.X("labels:Q", title="Number of hand labels"),
        y=alt.Y("width:Q", title="Confidence interval width (narrower is better)"),
        color=alt.Color("method:N", title="Method",
                        scale=alt.Scale(domain=["Debiased", "Labels only"], range=[BLUE, RED])),
        tooltip=["labels", "method", alt.Tooltip("width:Q", format=".3f")],
    ).properties(height=300)
    st.altair_chart(line, width="stretch")
    st.markdown(
        "The gap between the lines is the value the AI adds. When the AI is "
        "accurate the debiased line sits well below labels only. As the AI gets "
        "noisier the lines converge, which is the paper's warning: a weak "
        "predictor adds little, and you are better off just labelling more."
    )


# ---------------------------------------------------------------------------
# Step 10: report
# ---------------------------------------------------------------------------

st.header("Step 10. Export the report")


def build_report() -> str:
    L = ["# Debiased AI Signal report", "",
         f"Data source: {source}", f"Corpus size: {n_rows}",
         f"Validation sample: {n_labeled}", f"Confidence level: {int(conf * 100)} percent",
         f"AI scoring column: {prompt}", "",
         "## Average hawkishness",
         f"Naive: {nm['estimate']:+.3f} [{nm['low']:+.3f}, {nm['high']:+.3f}]",
         f"Labels only: {lm['estimate']:+.3f} [{lm['low']:+.3f}, {lm['high']:+.3f}]",
         f"Debiased: {pm['estimate']:+.3f} [{pm['low']:+.3f}, {pm['high']:+.3f}]"]
    if true_mean_ref is not None:
        L.append(f"Known truth: {true_mean_ref:+.3f}")
    L.append(f"Debiased mean is worth about {neff_mean:.0f} hand labels.")
    if has_outcome and po is not None:
        L += ["", "## Effect of hawkishness on the outcome",
              f"Naive slope: {no['estimate']:+.3f} [{no['low']:+.3f}, {no['high']:+.3f}]",
              f"Labels only slope: {lo['estimate']:+.3f} [{lo['low']:+.3f}, {lo['high']:+.3f}]",
              f"Debiased slope: {po['estimate']:+.3f} [{po['low']:+.3f}, {po['high']:+.3f}]"]
        if beta_ref is not None:
            L.append(f"Known truth: {beta_ref:+.3f}")
    L += ["", "## Method",
          "Debiasing follows Dell and Rambachan (2026), NBER 35744, framing AI "
          "measurement as a missing data problem. A random validation sample "
          "estimates how the AI errs and corrects the downstream estimate. The "
          "correction is valid even when the AI is biased; a more accurate AI only "
          "tightens the interval."]
    if mode == "real":
        L += ["", "Real data: FOMC-RoBERTa scores over the Trillion Dollar Words "
              "dataset, with the dataset's human labels as truth."]
    return "\n".join(L)


report_text = build_report()
st.download_button("Download the one page report (Markdown)", data=report_text,
                   file_name="debiased_signal_report.md", mime="text/markdown")
with st.expander("Preview the report"):
    st.code(report_text, language="markdown")

st.divider()
st.caption(
    "Method: Dell and Rambachan (2026), The Measurement Revolution, NBER Working "
    "Paper 35744. Mean case coincides with prediction powered inference "
    "(Angelopoulos et al., 2023). Real signal: gtfintechlab FOMC-RoBERTa and the "
    "Trillion Dollar Words dataset (Shah, Paturi, Chava, ACL 2023). For research "
    "demonstration."
)

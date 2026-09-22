"""
Debiasing AI Signals: a live walkthrough.

A small tool that shows why raw AI scores can mislead you, and how a small hand
labelled sample fixes it. Built around Dell and Rambachan (2026), NBER 35744.

Everything on screen is computed live from the data. The demonstration corpus is
simulated so the effect is visible, but the statistics are real, and you can
upload your own scored data to run it for real.
"""

from __future__ import annotations

import io
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

DATA_PATH = Path(__file__).parent / "data" / "fomc_demo.csv"
PROMPT_COLS = ["ai_promptA", "ai_promptB", "ai_promptC"]
PROMPT_LABELS = {
    "ai_promptA": "Prompt A (reads slightly more hawkish)",
    "ai_promptB": "Prompt B (reads slightly more dovish)",
    "ai_promptC": "Prompt C (fairly balanced)",
}
TRUE_MEAN_KNOWN = None   # filled after load, only for the demo
BETA_TRUE_KNOWN = 5.0    # the demo was generated with this slope

st.set_page_config(
    page_title="Debiasing AI Signals",
    page_icon="\U0001F4CF",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data
def load_demo() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def draw_labeled(n_rows: int, n_labeled: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_rows)
    return np.sort(idx[:n_labeled]), np.sort(idx[n_labeled:])


def ci_chart(rows, value_title: str, truth: float | None = None, unit: str = ""):
    """Horizontal point and interval chart, one row per method."""
    data = pd.DataFrame(rows)
    order = list(data["method"])
    base = alt.Chart(data)

    rule = base.mark_rule(strokeWidth=3, color="#1f4e79").encode(
        y=alt.Y("method:N", sort=order, title=None),
        x=alt.X("low:Q", title=f"{value_title} {unit}".strip()),
        x2="high:Q",
    )
    point = base.mark_point(size=140, filled=True, color="#1f4e79").encode(
        y=alt.Y("method:N", sort=order),
        x="estimate:Q",
        tooltip=[
            alt.Tooltip("method:N", title="Method"),
            alt.Tooltip("estimate:Q", title="Estimate", format=".3f"),
            alt.Tooltip("low:Q", title="Low", format=".3f"),
            alt.Tooltip("high:Q", title="High", format=".3f"),
        ],
    )
    label = base.mark_text(align="left", dx=10, dy=-12, color="#1f4e79").encode(
        y=alt.Y("method:N", sort=order),
        x="estimate:Q",
        text=alt.Text("estimate:Q", format=".2f"),
    )
    layers = [rule, point, label]

    if truth is not None:
        tdf = pd.DataFrame({"truth": [truth]})
        tline = alt.Chart(tdf).mark_rule(
            strokeDash=[6, 4], color="#c0392b", strokeWidth=2
        ).encode(x="truth:Q")
        layers.append(tline)

    return alt.layer(*layers).properties(height=180)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Controls")

source = st.sidebar.radio(
    "Data source",
    ["Built in demo (Fed communications)", "Upload your own CSV"],
    help="The demo works with no setup. Upload requires an AI score column, a "
         "true label column (blank where not labelled), and an outcome column.",
)

uploaded = None
if source == "Upload your own CSV":
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

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
# Load and prepare
# ---------------------------------------------------------------------------

demo_mode = source == "Built in demo (Fed communications)"

if demo_mode:
    df = load_demo().copy()
    TRUE_MEAN_KNOWN = float(df["true_hawk"].mean())
    available_prompts = PROMPT_COLS
    outcome_col = "yield_change"
    true_col = "true_hawk"
else:
    if uploaded is None:
        st.info("Upload a CSV in the sidebar, or switch back to the built in demo.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")
        st.stop()

    st.sidebar.markdown("Map your columns")
    cols = list(df.columns)
    score_col = st.sidebar.selectbox("AI score column", cols)
    true_col = st.sidebar.selectbox("True label column (blank where unlabelled)", cols)
    outcome_col = st.sidebar.selectbox("Outcome column", cols)
    df = df.rename(columns={score_col: "ai_uploaded"})
    available_prompts = ["ai_uploaded"]
    PROMPT_LABELS["ai_uploaded"] = "Your uploaded AI score"
    TRUE_MEAN_KNOWN = None
    BETA_TRUE_KNOWN = None

n_rows = len(df)


# ---------------------------------------------------------------------------
# Title and framing
# ---------------------------------------------------------------------------

st.title("Debiasing AI Signals")
st.markdown(
    "A raw AI score is a measurement with unknown error, not a fact. Feed it "
    "straight into a regression and your answer can be biased and your "
    "confidence interval too narrow, so you end up confidently wrong. This tool "
    "shows the problem and the fix from Dell and Rambachan (2026): hand label a "
    "small random sample, learn how the AI errs, and correct the final estimate. "
    "The correction stays valid even when the AI is poor."
)

if demo_mode:
    st.caption(
        "Demonstration mode. The corpus below is simulated so the effect is easy "
        "to see, and the true answer is known, so you can check that the method "
        "recovers it. Every statistic is computed live. Switch to upload in the "
        "sidebar to run this on your own scored data."
    )


# ---------------------------------------------------------------------------
# Step 1: Load the text
# ---------------------------------------------------------------------------

st.header("Step 1. Load the corpus")
st.write(
    f"The corpus has {n_rows} items. In a real run these are documents. Here we "
    "already have a numeric score per item plus the market outcome on that date."
)
st.dataframe(df.head(8), width="stretch")


# ---------------------------------------------------------------------------
# Step 2: Rubric
# ---------------------------------------------------------------------------

st.header("Step 2. Write the rubric (the one step that needs your judgement)")
st.write(
    "Pin down what you are measuring before you look at any result, so you cannot "
    "quietly tune the definition to get the answer you want. This is illustrative "
    "text here, but it is the anchor the whole method rests on."
)
st.text_area(
    "Hawkishness rubric",
    "Score each communication from minus 1 to plus 1.\n"
    "Plus 1: strongly leans toward tighter policy or names inflation as the main risk.\n"
    "Zero: balanced, no clear lean.\n"
    "Minus 1: strongly leans toward easing or names growth and jobs as the main risk.\n"
    "Edge case: naming inflation risk while holding rates steady counts as mildly hawkish.",
    height=140,
)


# ---------------------------------------------------------------------------
# Step 3: AI scores
# ---------------------------------------------------------------------------

st.header("Step 3. The AI scores everything")
if demo_mode:
    st.write(
        "The same model was asked with three reasonable but different prompts. "
        "Pick which one to treat as your signal. The point of the exercise is that "
        "this choice should not change your final answer, but for the naive method "
        "it does."
    )
    prompt = st.selectbox(
        "AI scoring to use", available_prompts,
        format_func=lambda c: PROMPT_LABELS[c], index=2,
    )
else:
    prompt = available_prompts[0]
    st.write("Using your uploaded AI score column.")

score = df[prompt].to_numpy(dtype=float)
outcome = df[outcome_col].to_numpy(dtype=float)

hist = alt.Chart(pd.DataFrame({"score": score})).mark_bar(color="#1f4e79").encode(
    x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="AI hawkishness score"),
    y=alt.Y("count()", title="Number of items"),
).properties(height=200)
st.altair_chart(hist, width="stretch")


# ---------------------------------------------------------------------------
# Step 4: Hand label a random sample
# ---------------------------------------------------------------------------

st.header("Step 4. Hand label a small random sample")

if demo_mode:
    st.write(
        "Draw a random sample and score it yourself against the rubric. Here the "
        "careful human labels are already known, so we simply reveal them for the "
        "sampled rows. Nothing outside this sample ever uses the true value."
    )
    n_labeled = st.slider(
        "Number of items you hand label", min_value=20, max_value=500, value=150, step=10,
    )
    lab_idx, unlab_idx = draw_labeled(n_rows, n_labeled, seed)
    true_all = df[true_col].to_numpy(dtype=float)
    y_lab = true_all[lab_idx]
    f_lab = score[lab_idx]
    f_unlab = score[unlab_idx]
    o_lab = outcome[lab_idx]
    o_unlab = outcome[unlab_idx]
else:
    labeled_mask = df[true_col].notna().to_numpy()
    lab_idx = np.where(labeled_mask)[0]
    unlab_idx = np.where(~labeled_mask)[0]
    n_labeled = int(labeled_mask.sum())
    st.write(
        f"Your file has {n_labeled} rows with a true label and "
        f"{len(unlab_idx)} without. The labelled rows are the validation sample."
    )
    true_all = df[true_col].to_numpy(dtype=float)
    y_lab = true_all[lab_idx]
    f_lab = score[lab_idx]
    f_unlab = score[unlab_idx]
    o_lab = outcome[lab_idx]
    o_unlab = outcome[unlab_idx]

if len(lab_idx) < 5 or len(unlab_idx) < 5:
    st.warning("Need at least a few labelled and a few unlabelled rows to proceed.")
    st.stop()

st.caption(
    f"Validation sample: {len(lab_idx)} items. Corpus scored by AI: {n_rows} items."
)


# ---------------------------------------------------------------------------
# Step 5 and 7: correct, then compare the mean
# ---------------------------------------------------------------------------

st.header("Step 5. Correct, and see the two answers side by side")
st.write(
    "First estimand: the average hawkishness across the corpus. Three ways to get "
    "it. Naive trusts the AI. Labels only ignores the AI and uses just your "
    "sample. Debiased uses both."
)

nm = naive_mean(score, alpha=alpha)
lm = labeled_only_mean(y_lab, alpha=alpha)
pm = ppi_mean(y_lab, f_lab, f_unlab, alpha=alpha)

mean_rows = [
    {"method": "Naive (trust the AI)", **{k: nm[k] for k in ("estimate", "low", "high")}},
    {"method": "Labels only", **{k: lm[k] for k in ("estimate", "low", "high")}},
    {"method": "Debiased", **{k: pm[k] for k in ("estimate", "low", "high")}},
]
st.altair_chart(
    ci_chart(mean_rows, "Average hawkishness", truth=TRUE_MEAN_KNOWN),
    width="stretch",
)
if demo_mode:
    st.markdown(
        f"How to read this. The red dashed line is the known truth "
        f"({TRUE_MEAN_KNOWN:+.3f}). The naive estimate misses it and its interval "
        f"is tight, so it looks more certain than it is. The debiased estimate "
        f"lands on the truth. Labels only is also honest but wider, because it "
        f"throws away the corpus."
    )


# ---------------------------------------------------------------------------
# Step 6 and 7: downstream question, the regression
# ---------------------------------------------------------------------------

st.header("Step 6. Ask the downstream question")
st.write(
    "Does a more hawkish communication move the market on the day? We regress the "
    "outcome on hawkishness. Here the AI score is the regressor, which is the "
    "harder errors in variables case, and the debiasing corrects for it."
)

no = naive_ols(outcome, score, alpha=alpha)
lo = labeled_only_ols(o_lab, y_lab, alpha=alpha)
po = ppi_ols(o_lab, y_lab, f_lab, o_unlab, f_unlab, alpha=alpha)

slope_rows = [
    {"method": "Naive (trust the AI)", **{k: no[k] for k in ("estimate", "low", "high")}},
    {"method": "Labels only", **{k: lo[k] for k in ("estimate", "low", "high")}},
    {"method": "Debiased", **{k: po[k] for k in ("estimate", "low", "high")}},
]
st.altair_chart(
    ci_chart(slope_rows, "Effect on the outcome per unit of hawkishness",
             truth=BETA_TRUE_KNOWN, unit=""),
    width="stretch",
)
if demo_mode:
    st.markdown(
        f"How to read this. The truth is {BETA_TRUE_KNOWN:.1f}. The naive slope is "
        f"pulled toward zero because the AI score is noisy, so it understates how "
        f"much the market responds. The debiased slope recovers the true effect."
    )


# ---------------------------------------------------------------------------
# Step 8: stress test the prompt
# ---------------------------------------------------------------------------

if demo_mode:
    st.header("Step 8. Stress test the prompt")
    st.write(
        "Run the same rubric under all three prompts. Watch the naive answer swing "
        "with the wording while the debiased answer stays put. That stability is "
        "the whole point: your conclusion should not depend on how you phrased the "
        "prompt."
    )

    rows = []
    for c in PROMPT_COLS:
        sc = df[c].to_numpy(dtype=float)
        nm_c = naive_mean(sc, alpha=alpha)
        pm_c = ppi_mean(y_lab, sc[lab_idx], sc[unlab_idx], alpha=alpha)
        rows.append({"prompt": c.replace("ai_prompt", "Prompt "),
                     "kind": "Naive", **{k: nm_c[k] for k in ("estimate", "low", "high")}})
        rows.append({"prompt": c.replace("ai_prompt", "Prompt "),
                     "kind": "Debiased", **{k: pm_c[k] for k in ("estimate", "low", "high")}})
    sdf = pd.DataFrame(rows)

    err = alt.Chart(sdf).mark_rule(strokeWidth=3).encode(
        x=alt.X("prompt:N", title=None),
        y=alt.Y("low:Q", title="Average hawkishness"),
        y2="high:Q",
        color=alt.Color("kind:N", title="Method",
                        scale=alt.Scale(domain=["Naive", "Debiased"],
                                        range=["#c0392b", "#1f4e79"])),
        xOffset="kind:N",
    )
    pts = alt.Chart(sdf).mark_point(size=120, filled=True).encode(
        x=alt.X("prompt:N"), y="estimate:Q",
        color=alt.Color("kind:N",
                        scale=alt.Scale(domain=["Naive", "Debiased"],
                                        range=["#c0392b", "#1f4e79"])),
        xOffset="kind:N",
        tooltip=["prompt", "kind",
                 alt.Tooltip("estimate:Q", format=".3f")],
    )
    truth_line = alt.Chart(pd.DataFrame({"t": [TRUE_MEAN_KNOWN]})).mark_rule(
        strokeDash=[6, 4], color="#333").encode(y="t:Q")
    st.altair_chart((err + pts + truth_line).properties(height=320),
                    width="stretch")

    naive_range = sdf[sdf.kind == "Naive"]["estimate"]
    deb_range = sdf[sdf.kind == "Debiased"]["estimate"]
    st.markdown(
        f"The naive estimate ranges over {naive_range.max() - naive_range.min():.3f} "
        f"across the three prompts. The debiased estimate ranges over only "
        f"{deb_range.max() - deb_range.min():.3f}, clustered on the truth."
    )


# ---------------------------------------------------------------------------
# Step 9: is it worth it
# ---------------------------------------------------------------------------

st.header("Step 9. Is it worth it")
st.write(
    "Debiasing borrows strength from the unlabelled corpus, so a handful of labels "
    "can carry the weight of many more. This shows how the confidence interval "
    "narrows as you label more, and what your current sample is worth."
)

neff_mean = effective_sample_size(n_labeled, lm["se"] ** 2, pm["se"] ** 2)
neff_slope = effective_sample_size(n_labeled, lo["se"] ** 2, po["se"] ** 2)

c1, c2 = st.columns(2)
c1.metric("Labels you actually did", f"{n_labeled}")
c2.metric("Debiased mean is worth about", f"{neff_mean:.0f} hand labels")

grid = [g for g in range(20, min(n_rows - 5, 500) + 1, 20)]
curve = []
for g in grid:
    li, ui = draw_labeled(n_rows, g, seed)
    yl = true_all[li]
    fl = score[li]
    fu = score[ui]
    pm_g = ppi_mean(yl, fl, fu, alpha=alpha)
    lm_g = labeled_only_mean(yl, alpha=alpha)
    curve.append({"labels": g, "width": pm_g["high"] - pm_g["low"], "method": "Debiased"})
    curve.append({"labels": g, "width": lm_g["high"] - lm_g["low"], "method": "Labels only"})
cdf = pd.DataFrame(curve)

line = alt.Chart(cdf).mark_line(point=True).encode(
    x=alt.X("labels:Q", title="Number of hand labels"),
    y=alt.Y("width:Q", title="Confidence interval width (narrower is better)"),
    color=alt.Color("method:N", title="Method",
                    scale=alt.Scale(domain=["Debiased", "Labels only"],
                                    range=["#1f4e79", "#c0392b"])),
    tooltip=["labels", "method", alt.Tooltip("width:Q", format=".3f")],
).properties(height=300)
st.altair_chart(line, width="stretch")
st.markdown(
    "The gap between the two lines is the value the AI adds. When the AI is "
    "accurate the debiased line sits well below labels only. As the AI gets "
    "noisier the two lines converge, which is the paper's warning: a weak "
    "predictor adds little, and you are better off just labelling more."
)


# ---------------------------------------------------------------------------
# Step 10: export the report
# ---------------------------------------------------------------------------

st.header("Step 10. Export the report")


def build_report() -> str:
    lines = []
    lines.append("# Debiased AI Signal report")
    lines.append("")
    lines.append(f"Corpus size: {n_rows}")
    lines.append(f"Validation sample: {n_labeled}")
    lines.append(f"Confidence level: {int(conf * 100)} percent")
    lines.append(f"AI scoring column: {prompt}")
    lines.append("")
    lines.append("## Average hawkishness")
    lines.append(f"Naive: {nm['estimate']:+.3f} [{nm['low']:+.3f}, {nm['high']:+.3f}]")
    lines.append(f"Labels only: {lm['estimate']:+.3f} [{lm['low']:+.3f}, {lm['high']:+.3f}]")
    lines.append(f"Debiased: {pm['estimate']:+.3f} [{pm['low']:+.3f}, {pm['high']:+.3f}]")
    if TRUE_MEAN_KNOWN is not None:
        lines.append(f"Known truth (demo): {TRUE_MEAN_KNOWN:+.3f}")
    lines.append(f"Debiased mean is worth about {neff_mean:.0f} hand labels.")
    lines.append("")
    lines.append("## Effect of hawkishness on the outcome")
    lines.append(f"Naive slope: {no['estimate']:+.3f} [{no['low']:+.3f}, {no['high']:+.3f}]")
    lines.append(f"Labels only slope: {lo['estimate']:+.3f} [{lo['low']:+.3f}, {lo['high']:+.3f}]")
    lines.append(f"Debiased slope: {po['estimate']:+.3f} [{po['low']:+.3f}, {po['high']:+.3f}]")
    if BETA_TRUE_KNOWN is not None:
        lines.append(f"Known truth (demo): {BETA_TRUE_KNOWN:+.3f}")
    lines.append("")
    lines.append("## Method")
    lines.append(
        "Debiasing follows Dell and Rambachan (2026), NBER 35744, framing AI "
        "measurement as a missing data problem. A random validation sample "
        "estimates how the AI errs and corrects the downstream estimate. The "
        "correction is valid even when the AI is biased; a more accurate AI only "
        "tightens the interval."
    )
    return "\n".join(lines)


report_text = build_report()
st.download_button(
    "Download the one page report (Markdown)",
    data=report_text,
    file_name="debiased_signal_report.md",
    mime="text/markdown",
)
with st.expander("Preview the report"):
    st.code(report_text, language="markdown")

st.divider()
st.caption(
    "Method: Dell and Rambachan (2026), The Measurement Revolution, NBER Working "
    "Paper 35744. Mean case coincides with prediction powered inference "
    "(Angelopoulos et al., 2023). Built for research demonstration."
)

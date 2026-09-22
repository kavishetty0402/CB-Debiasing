"""
Debiasing estimators for AI-generated measurements.

This module implements the correction described in Dell and Rambachan (2026),
"The Measurement Revolution? Credible Measurement and Inference in the Age of AI"
(NBER Working Paper 35744), which frames inference with AI predictions as a
missing data problem (their MAR-S framework). The mean case coincides with the
prediction-powered inference estimator of Angelopoulos et al. (2023).

The idea in one line: an AI scores every document, you hand label a small random
sample, and the sample tells you how the AI errs so you can correct the final
estimate. The correction stays valid even when the AI is badly biased. A better
AI only buys you a tighter confidence interval, not a more correct answer.

Two estimands are provided:
  1. a population mean (for example, average hawkishness across meetings)
  2. an OLS slope where the AI scored variable is a regressor (for example,
     the effect of hawkishness on the change in the 2 year yield)

In the OLS case the imputed variable is on the right hand side, which is the
errors in variables setting. The general estimating equation form used here
corrects for that, unlike the classic prediction-powered inference routine that
assumes the outcome is imputed.

All statistics are computed live. Nothing here is faked.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _z(alpha: float) -> float:
    """Two sided normal critical value."""
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


# ---------------------------------------------------------------------------
# Mean estimands
# ---------------------------------------------------------------------------

def naive_mean(pred_all, alpha: float = 0.05):
    """Treat every AI score as if it were the truth and take the average.

    This is what most pipelines do. It is biased whenever the AI is biased,
    and its confidence interval is too narrow because it ignores that error.
    """
    p = np.asarray(pred_all, dtype=float)
    n = p.size
    theta = p.mean()
    se = p.std(ddof=1) / np.sqrt(n)
    half = _z(alpha) * se
    return {"estimate": theta, "se": se, "low": theta - half, "high": theta + half}


def labeled_only_mean(y_lab, alpha: float = 0.05):
    """Use only the hand labels and ignore the AI entirely.

    This is unbiased but wasteful, because it throws away the large unlabeled
    corpus. It is the honest but low power baseline.
    """
    y = np.asarray(y_lab, dtype=float)
    n = y.size
    theta = y.mean()
    se = y.std(ddof=1) / np.sqrt(n)
    half = _z(alpha) * se
    return {"estimate": theta, "se": se, "low": theta - half, "high": theta + half}


def ppi_mean(y_lab, f_lab, f_unlab, alpha: float = 0.05):
    """Debiased mean using the hand labels to correct the AI scores.

    y_lab   true labels on the hand labelled sample
    f_lab   AI scores on the hand labelled sample
    f_unlab AI scores on everything that was not hand labelled

    Estimator: mean(f on unlabelled) + mean(true minus AI on labelled).
    The first term is the cheap AI based estimate. The second term is the
    correction learned from the labelled sample.
    """
    y_lab = np.asarray(y_lab, dtype=float)
    f_lab = np.asarray(f_lab, dtype=float)
    f_unlab = np.asarray(f_unlab, dtype=float)

    n = y_lab.size
    big_n = f_unlab.size

    rectifier = y_lab - f_lab
    theta = f_unlab.mean() + rectifier.mean()
    var = f_unlab.var(ddof=1) / big_n + rectifier.var(ddof=1) / n
    se = np.sqrt(var)
    half = _z(alpha) * se
    return {"estimate": theta, "se": se, "low": theta - half, "high": theta + half}


# ---------------------------------------------------------------------------
# OLS estimands with an AI scored regressor
# ---------------------------------------------------------------------------

def _design(x):
    """Build an intercept plus single regressor design matrix."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.column_stack([np.ones_like(x), x])


def naive_ols(y_all, f_all, alpha: float = 0.05):
    """Regress the outcome on the AI score, treating the score as truth.

    Returns the slope and a heteroskedasticity robust (HC1) interval.
    """
    y = np.asarray(y_all, dtype=float).reshape(-1)
    X = _design(f_all)
    n, k = X.shape

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    scale = n / (n - k)
    cov = scale * (xtx_inv @ meat @ xtx_inv)

    slope = beta[1]
    se = np.sqrt(cov[1, 1])
    half = _z(alpha) * se
    return {"estimate": slope, "se": se, "low": slope - half, "high": slope + half,
            "intercept": beta[0]}


def labeled_only_ols(y_lab, x_true_lab, alpha: float = 0.05):
    """Regress the outcome on the true hand labelled score, labelled rows only.

    Unbiased but uses only the small labelled set, so the interval is wide.
    """
    y = np.asarray(y_lab, dtype=float).reshape(-1)
    X = _design(x_true_lab)
    n, k = X.shape

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    scale = n / (n - k)
    cov = scale * (xtx_inv @ meat @ xtx_inv)

    slope = beta[1]
    se = np.sqrt(cov[1, 1])
    half = _z(alpha) * se
    return {"estimate": slope, "se": se, "low": slope - half, "high": slope + half,
            "intercept": beta[0]}


def ppi_ols(y_lab, x_true_lab, f_lab, y_unlab, f_unlab, alpha: float = 0.05):
    """Debiased OLS slope when the regressor is AI scored.

    The outcome (for example the yield change) is observed for every row. The
    regressor (hawkishness) is AI scored everywhere and hand labelled only on
    the validation sample.

    Solves the corrected estimating equation
        mean over unlabelled of  x_hat (y minus x_hat beta)
      + mean over labelled of   [ x_true (y minus x_true beta)
                                  minus x_hat (y minus x_hat beta) ]  = 0
    which is linear in beta, so it has a closed form. The variance uses the
    standard sandwich for this two sample estimating equation.
    """
    y_lab = np.asarray(y_lab, dtype=float).reshape(-1)
    y_unlab = np.asarray(y_unlab, dtype=float).reshape(-1)

    Xt = _design(x_true_lab)   # true regressor, labelled rows
    Xi = _design(f_lab)        # AI regressor, labelled rows
    Xu = _design(f_unlab)      # AI regressor, unlabelled rows

    n = Xt.shape[0]
    big_n = Xu.shape[0]

    a_unlab = (Xu.T @ Xu) / big_n
    b_unlab = (Xu.T @ y_unlab) / big_n

    a_lab_true = (Xt.T @ Xt) / n
    b_lab_true = (Xt.T @ y_lab) / n

    a_lab_imp = (Xi.T @ Xi) / n
    b_lab_imp = (Xi.T @ y_lab) / n

    M = a_unlab + a_lab_true - a_lab_imp
    v = b_unlab + b_lab_true - b_lab_imp
    M_inv = np.linalg.inv(M)
    beta = M_inv @ v

    # sandwich variance from the two independent sample groups
    g_unlab = Xu * (y_unlab - Xu @ beta)[:, None]
    g_true = Xt * (y_lab - Xt @ beta)[:, None]
    g_imp = Xi * (y_lab - Xi @ beta)[:, None]
    g_diff = g_true - g_imp

    s_u = np.cov(g_unlab, rowvar=False, ddof=1)
    s_l = np.cov(g_diff, rowvar=False, ddof=1)

    omega = s_u / big_n + s_l / n
    cov = M_inv @ omega @ M_inv.T

    slope = beta[1]
    se = np.sqrt(cov[1, 1])
    half = _z(alpha) * se
    return {"estimate": slope, "se": se, "low": slope - half, "high": slope + half,
            "intercept": beta[0]}


# ---------------------------------------------------------------------------
# Efficiency: how many pure hand labels would match the debiased precision
# ---------------------------------------------------------------------------

def effective_sample_size(n_labeled: int, var_labeled_only: float, var_ppi: float) -> float:
    """Translate the precision gain into an intuitive number.

    The debiased estimate has variance var_ppi using n_labeled hand labels plus
    the AI. The hand labels alone would need this many observations to match it.
    """
    if var_ppi <= 0:
        return float("nan")
    return float(n_labeled * var_labeled_only / var_ppi)

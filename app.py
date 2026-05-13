"""Streamlit demo: TCNL Quanta standardised-test composite held-out prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Quanta ST-composite prediction",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def load_predictions() -> tuple[pd.DataFrame, list[str]]:
    npz = np.load(DATA / "holdout_predictions.npz", allow_pickle=True)
    names = list(npz["target_names"])
    row = npz["subject"].astype(str)  # public artifacts use position labels row-NNN
    is_test = npz["is_test"].astype(bool)
    fold = npz["fold_ids"].astype(int)
    y = npz["y_true"]
    cv = npz["cv_oof"]
    te = npz["test_pred"]
    rows = []
    for ti, name in enumerate(names):
        pred = np.where(is_test, te[:, ti], cv[:, ti])
        rows.append(
            pd.DataFrame(
                {
                    "row": row,
                    "target": name,
                    "y_true": y[:, ti],
                    "pred": pred,
                    "fold": fold,
                    "is_test": is_test,
                    "split": np.where(is_test, "Held-out test (n=86)", "Train CV OOF (n=341)"),
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["err"] = df["pred"] - df["y_true"]
    df["abs_err"] = df["err"].abs()
    return df, names


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(DATA / "holdout_summary.csv")


@st.cache_data
def load_ablation() -> pd.DataFrame:
    return pd.read_csv(DATA / "modality_ablation.csv")


@st.cache_data
def load_deviation_corr() -> pd.DataFrame:
    return pd.read_csv(DATA / "deviation_correlations.csv")


@st.cache_data
def load_deviation_zhang() -> tuple[pd.DataFrame, list[str]]:
    npz = np.load(DATA / "deviation_zhang.npz", allow_pickle=True)
    names = list(npz["target_names"])
    row = npz["subject"].astype(str)  # public artifacts use position labels row-NNN
    is_test = npz["is_test"].astype(bool)
    fold = npz["fold_ids"].astype(int)
    rows = []
    for ti, name in enumerate(names):
        rows.append(
            pd.DataFrame(
                {
                    "row": row,
                    "target": name,
                    "y_true": npz["y_true"][:, ti],
                    "pred": npz["pred_full"][:, ti],
                    "deviation_zhang": npz["deviation"][:, ti],
                    "fold": fold,
                    "is_test": is_test,
                    "split": np.where(is_test, "Held-out test (n=86)", "Train CV OOF (n=341)"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), names


@st.cache_data
def load_meta() -> dict:
    return json.loads((DATA / "st_targets_meta.json").read_text())


preds, target_names = load_predictions()
summary = load_summary()
ablation = load_ablation()
dev_corr = load_deviation_corr()
zhang, _ = load_deviation_zhang()
meta = load_meta()


SHORT = {
    "LANGUAGE_ST_NORM_VCI": "Language — VCI",
    "MEMORY_ST_NORM_AudImm": "Memory — Auditory Immediate",
    "MEMORY_ST_NORM_VisImm": "Memory — Visual Immediate",
    "MEMORY_ST_NORM_WorMem": "Memory — Working Memory",
    "MOTOR_ST_SCALED_FineMotor": "Motor — Fine Motor",
    "MOTOR_ST_SCALED_Balance": "Motor — Balance",
    "MOTOR_ST_SCALED_ProcessingSpeed": "Motor — Processing Speed",
}
target_options = [(SHORT.get(n, n), n) for n in target_names]


st.title("Quanta — predicting standardised-test composites from brain & behaviour")
st.caption(
    "Frozen 80/20 split (n_train=341, n_test=86), inherited from the Quanta brain-age "
    "held-out repo. Seven composite targets, seven modality blocks, per-block RidgeCV → "
    "RidgeCV meta-stack. Use the sidebar to pick a target; tabs drill into the prediction, "
    "modality contribution, and external-validity (Zhang-corrected deviation) views."
)

with st.sidebar:
    st.header("Target composite")
    label_to_name = {lbl: n for (lbl, n) in target_options}
    chosen_label = st.radio(
        "Select",
        list(label_to_name.keys()),
        label_visibility="collapsed",
    )
    chosen = label_to_name[chosen_label]
    st.markdown("---")
    st.markdown(
        f"**Cohort.** {meta['n_subjects']} healthy adults, frozen 80/20 split "
        f"({meta['n_train']} train / {meta['n_test']} test)."
    )
    st.markdown(
        "**Dropped composites** (linear in kept targets):  \n"
        + "  \n".join([f"`{k}` — {v}" for k, v in meta["dropped_candidates"].items()])
    )

# ── headline ─────────────────────────────────────────────────────────────────
row = summary[summary["target"] == chosen].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Train-CV MAE", f"{row['cv_mae']:.2f}", f"± {row['cv_mae_std']:.2f}")
c2.metric("Held-out MAE", f"{row['test_mae']:.2f}", f"vs mean-baseline {row['test_mae']-row['test_mae_baseline']:+.2f}")
c3.metric("Held-out R²", f"{row['test_r2']:.3f}", f"baseline R² {row['test_r2_baseline']:+.3f}")
c4.metric("Held-out Pearson ρ", f"{row['test_pearson']:.3f}")
c5.metric("n_test with y", f"{int(row['n_test_with_y'])}/86")

tabs = st.tabs([
    "Predicted vs true",
    "Per-target overview",
    "Modality ablation",
    "Deviation correlations",
    "About",
])

# ── tab 1 ────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Predicted vs true — {chosen_label}")
    sub = preds[preds["target"] == chosen].dropna(subset=["y_true", "pred"]).copy()
    lo = float(min(sub["y_true"].min(), sub["pred"].min())) - 3
    hi = float(max(sub["y_true"].max(), sub["pred"].max())) + 3
    fig = px.scatter(
        sub, x="y_true", y="pred", color="split", symbol="split",
        hover_data={"fold": True, "abs_err": ":.2f", "is_test": False, "split": False,
                    "y_true": ":.1f", "pred": ":.1f", "row": True},
        labels={"y_true": "True composite score", "pred": "Predicted composite score"},
        color_discrete_map={
            "Train CV OOF (n=341)": "#7aa6c2",
            "Held-out test (n=86)": "#d2766b",
        },
        opacity=0.75,
    )
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#888", dash="dash"), name="identity", showlegend=False,
    ))
    fig.update_layout(
        height=540,
        xaxis=dict(range=[lo, hi], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[lo, hi]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

# ── tab 2 ────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("All seven composite targets")
    df = summary.copy()
    df["target_label"] = df["target"].map(SHORT).fillna(df["target"])
    df_disp = df[["target_label", "n_train", "n_test_with_y",
                  "cv_mae", "cv_mae_std", "test_mae", "test_rmse",
                  "test_r2", "test_pearson", "test_mae_baseline", "test_r2_baseline"]].copy()
    for c in df_disp.columns:
        if df_disp[c].dtype.kind == "f":
            df_disp[c] = df_disp[c].map(lambda v: f"{v:.3f}")
    df_disp.columns = ["Target", "n train", "n test", "CV MAE", "CV SD",
                       "Test MAE", "Test RMSE", "Test R²", "Test ρ",
                       "Baseline MAE", "Baseline R²"]
    st.dataframe(df_disp, hide_index=True, width="stretch")

    fig = px.bar(
        df.sort_values("test_r2"),
        x="test_r2", y="target_label", orientation="h",
        text=df.sort_values("test_r2")["test_r2"].map(lambda v: f"{v:.3f}"),
        color="test_r2", color_continuous_scale="Blues",
        labels={"test_r2": "Held-out R²", "target_label": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=10, r=10))
    st.plotly_chart(fig, width="stretch")

# ── tab 3 ────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader(f"Modality block ablation — {chosen_label}")
    st.caption(
        "Δ MAE relative to the full 7-block stack. Positive bars = removing that block "
        "hurts the fit (the block carried independent signal)."
    )
    a = ablation[ablation["target"] == chosen].copy()
    a = a.sort_values("delta_test_mae")
    fig = px.bar(
        a, x="delta_test_mae", y="dropped_block", orientation="h",
        text=a["delta_test_mae"].map(lambda v: f"{v:+.3f}"),
        labels={"delta_test_mae": "Δ held-out MAE", "dropped_block": ""},
        color="delta_test_mae", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Per-block CV impact**")
    cv_fig = px.bar(
        a.sort_values("delta_cv_mae"),
        x="delta_cv_mae", y="dropped_block", orientation="h",
        text=a.sort_values("delta_cv_mae")["delta_cv_mae"].map(lambda v: f"{v:+.3f}"),
        labels={"delta_cv_mae": "Δ training-CV MAE", "dropped_block": ""},
        color="delta_cv_mae", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
    )
    cv_fig.update_traces(textposition="outside")
    cv_fig.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(cv_fig, width="stretch")

    with st.expander("Full ablation table (all targets × blocks)"):
        st.dataframe(ablation, hide_index=True, width="stretch")

# ── tab 4 ────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader(f"Zhang-corrected deviation correlations — {chosen_label}")
    st.caption(
        "Per-subject (pred − bias-corrected reference) deviation correlated against the "
        "30 BASIC_Q_* questionnaire columns and the 70 unused _ST_ measures. "
        "Significance flag uses Benjamini-Hochberg FDR across the 100 features tested per target."
    )

    z = zhang[zhang["target"] == chosen].dropna(subset=["deviation_zhang", "y_true"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            z, x="y_true", y="deviation_zhang", color="split", symbol="split",
            hover_data={"fold": True, "row": True, "split": False},
            labels={"y_true": "True composite score", "deviation_zhang": "Zhang-corrected deviation"},
            color_discrete_map={
                "Train CV OOF (n=341)": "#7aa6c2",
                "Held-out test (n=86)": "#d2766b",
            },
            opacity=0.75,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#777")
        fig.update_layout(height=420, margin=dict(l=10, r=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig2 = px.histogram(
            z, x="deviation_zhang", color="split", nbins=40, opacity=0.7,
            color_discrete_map={
                "Train CV OOF (n=341)": "#7aa6c2",
                "Held-out test (n=86)": "#d2766b",
            },
            labels={"deviation_zhang": "Zhang-corrected deviation"},
        )
        fig2.update_layout(height=420, margin=dict(l=10, r=10))
        st.plotly_chart(fig2, width="stretch")

    corr = dev_corr[dev_corr["target"] == chosen].copy()
    corr["sig"] = corr["q_fdr"] < 0.05
    pick = st.radio("Filter", ["FDR-significant", "All"], horizontal=True)
    filt = corr[corr["sig"]] if pick == "FDR-significant" else corr
    cat_pick = st.multiselect(
        "Category",
        sorted(corr["category"].unique()),
        default=sorted(corr["category"].unique()),
    )
    filt = filt[filt["category"].isin(cat_pick)].sort_values("pearson_r")

    fig3 = px.bar(
        filt, x="pearson_r", y="feature", orientation="h",
        color="category",
        text=filt["pearson_r"].map(lambda v: f"{v:+.2f}"),
        labels={"pearson_r": "Pearson r vs Zhang-corrected deviation", "feature": ""},
        color_discrete_map={"Q": "#6e9a7a", "ST": "#a47bb5"},
    )
    fig3.update_layout(height=max(360, 22 * len(filt)), margin=dict(l=10, r=10))
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, width="stretch")

    disp = filt[["feature", "category", "n", "pearson_r", "ci_lo", "ci_hi", "p_raw", "q_fdr"]].copy()
    for c in ("pearson_r", "ci_lo", "ci_hi"):
        disp[c] = disp[c].map(lambda v: f"{v:+.3f}")
    for c in ("p_raw", "q_fdr"):
        disp[c] = disp[c].map(lambda v: f"{v:.3g}")
    disp.columns = ["feature", "category", "n", "r", "CI low", "CI high", "p", "q (FDR)"]
    with st.expander("Correlation table"):
        st.dataframe(disp, hide_index=True, width="stretch")

# ── tab 5 ────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("About this demo")
    st.markdown(
        """
        **Cohort.** 427 healthy adults (age 19–80) from the TCNL Quanta multimodal
        ageing study. Phenotype includes behavioural batteries, resting EEG, T1-w
        MRI morphometry, DTI ROI (JHU-ICBM), resting-state fMRI Schaefer-400
        functional connectivity, and pretrained deep-feature scalars (SFCN +
        Pyment).

        **Targets.** Seven composite scores from the standardized cognitive and
        motor batteries — drawn from the test-publisher canonical
        norm/scaled indices. Targets that were exact or near-exact linear
        combinations of the kept set were dropped before modelling
        (`MEMORY_ST_NORM_ImmMem`, `LANGUAGE_ST_SCALED_SUM`,
        `LANGUAGE_ST_RAW_SUM`, `LANGUAGE_ST_NORM_PR`).

        **Model.** Per-block RidgeCV base learners produce out-of-fold
        predictions on the 341 train subjects and direct predictions on the 86
        held-out subjects. A RidgeCV meta-learner stacks the seven OOF columns
        into the final score. Each target is fit independently. No `_ST_` or
        `_Q_` feature is ever an input.

        **Split.** Inherited verbatim from the brain-age held-out repo: 80/20
        age × sex stratified, frozen ahead of the final run. Train-fitted
        means impute missing features. rs-fMRI FC is reduced to 50 PCs fit on
        the training subjects only.

        **What this demo is not.** No raw subject features ship with the repo.
        Only de-identified position-indexed prediction vectors, the run summary
        CSV/JSON, the modality ablation TSV, and the per-target Zhang-corrected
        deviation correlations are included.
        """
    )
    st.markdown(
        "**Source:** `tcnl2021-lab/quanta-st-composites` (preparing for public upload). "
        "Working repo: internal `Quanta_st_composites`."
    )

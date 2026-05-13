"""Streamlit demo: TCNL Quanta ST composite held-out prediction, three score levels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data"
LEVELS = ["RAW", "SCALED", "NORM"]
LEVEL_DESC = {
    "RAW": "Raw subtest sums — bounded by test ceilings, not age-normed.",
    "SCALED": "Age-normed scaled scores summed into composites.",
    "NORM": "Publisher norm-table indices on top of the scaled sums (non-linear lookup).",
}


st.set_page_config(page_title="Quanta ST-composite prediction", page_icon="🧠", layout="wide")


@st.cache_data
def load_predictions(level: str) -> tuple[pd.DataFrame, list[str]]:
    npz = np.load(DATA_ROOT / level.lower() / "holdout_predictions.npz", allow_pickle=True)
    names = list(npz["target_names"])
    row = npz["subject"].astype(str)
    is_test = npz["is_test"].astype(bool)
    fold = npz["fold_ids"].astype(int)
    y = npz["y_true"]; cv = npz["cv_oof"]; te = npz["test_pred"]
    rows = []
    for ti, name in enumerate(names):
        pred = np.where(is_test, te[:, ti], cv[:, ti])
        rows.append(pd.DataFrame({
            "row": row, "target": name,
            "y_true": y[:, ti], "pred": pred,
            "fold": fold, "is_test": is_test,
            "split": np.where(is_test, "Held-out test (n=86)", "Train CV OOF (n=341)"),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["err"] = df["pred"] - df["y_true"]
    df["abs_err"] = df["err"].abs()
    return df, names


@st.cache_data
def load_summary(level: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / level.lower() / "holdout_summary.csv")


@st.cache_data
def load_ablation(level: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / level.lower() / "modality_ablation.csv")


@st.cache_data
def load_deviation_corr(level: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / level.lower() / "deviation_correlations.csv")


@st.cache_data
def load_deviation(level: str) -> tuple[pd.DataFrame, list[str]]:
    npz = np.load(DATA_ROOT / level.lower() / "deviation.npz", allow_pickle=True)
    names = list(npz["target_names"])
    row = npz["subject"].astype(str)
    is_test = npz["is_test"].astype(bool)
    fold = npz["fold_ids"].astype(int)
    rows = []
    for ti, name in enumerate(names):
        rows.append(pd.DataFrame({
            "row": row, "target": name,
            "y_true": npz["y_true"][:, ti],
            "pred": npz["pred_full"][:, ti],
            "deviation": npz["deviation"][:, ti],
            "fold": fold, "is_test": is_test,
            "split": np.where(is_test, "Held-out test (n=86)", "Train CV OOF (n=341)"),
        }))
    return pd.concat(rows, ignore_index=True), names


@st.cache_data
def load_meta(level: str) -> dict:
    return json.loads((DATA_ROOT / level.lower() / "st_targets_meta.json").read_text())


@st.cache_data
def load_formulations(level: str) -> dict:
    return json.loads((DATA_ROOT / level.lower() / "composite_formulations.json").read_text())


_SHORT_TAIL = {
    "VCI": "VCI (composite)",
    "AudImm": "Auditory Imm. (composite)",
    "VisImm": "Visual Imm. (composite)",
    "WorMem": "Working Memory (composite)",
    "FineMotor": "Fine Motor (composite)",
    "Balance": "Balance (composite)",
    "ProcessingSpeed": "Processing Speed (composite)",
    "BPrestHigh": "BP rest, systolic", "BPrestLow": "BP rest, diastolic",
    "HRrest": "Heart rate, resting", "WHR": "Waist-hip ratio",
    "WaisSpeed": "WAIS Processing Speed",
    "GripMvcL": "Grip MVC (left)", "GripMvcR": "Grip MVC (right)",
    "LogMemI": "Logical Memory I", "FacI": "Faces I",
    "VerPairI": "Verbal Paired I", "FamPicI": "Family Pictures I",
    "LetNumSeq": "Letter-Number Seq.",
    "SpaForward": "Spatial Span Forward", "SpaBackward": "Spatial Span Backward",
    "SpaTotal": "Spatial Span Total",
    "BestLegEyeOpen": "Single leg, eyes open (best)",
    "WorseLegEyeOpen": "Single leg, eyes open (worse)",
    "BestLegEyeClose": "Single leg, eyes closed (best)",
    "WorseLegEyeClose": "Single leg, eyes closed (worse)",
    "FunctionalReaching": "Functional reaching",
    "PegboardDominant": "Pegboard (dominant)",
    "PegboardNondominant": "Pegboard (non-dominant)",
    "PegboardBoth": "Pegboard (both)",
    "PegboardAssemble": "Pegboard (assembly)",
    "SymbolSearch": "Symbol Search", "SymbolCoding": "Symbol Coding",
    "AerobicRating": "Aerobic rating",
    "SUM": "Verbal sum", "SIMILARITY": "Similarity",
    "VOCABULARY": "Vocabulary", "INFORMATION": "Information",
    "BMI": "BMI",
}
_DOMAIN = {"LANGUAGE": "Language", "MEMORY": "Memory", "MOTOR": "Motor"}


class _ShortLookup:
    def get(self, target, default=None):
        parts = target.split("_", 3)
        if len(parts) < 4 or parts[1] != "ST":
            return default if default is not None else target
        return f"{_DOMAIN.get(parts[0], parts[0])} — {_SHORT_TAIL.get(parts[3], parts[3])}"

    def __getitem__(self, target):
        return self.get(target, target)


SHORT = _ShortLookup()


st.title("Quanta — predicting standardised-test composites from brain & behaviour")
st.caption(
    "Frozen 80/20 split (n_train=341, n_test=86) inherited from the brain-age held-out "
    "repo. Seven modality blocks (BEH · EEG · MRI · DTI · DL · FC · SEX); chronological "
    "age is intentionally excluded. The model is fit independently at three score "
    "levels — RAW, SCALED, NORM — because the publisher's raw → scaled → norm "
    "transformation is non-linear. Pick a level and a target in the sidebar."
)

with st.sidebar:
    st.header("Score level")
    chosen_level = st.radio(
        "Level", LEVELS, label_visibility="collapsed",
        help="RAW = raw subtest sums. SCALED = age-normed scaled-subtest sums. "
             "NORM = publisher norm-table index on top of the scaled sum.",
    )
    st.caption(LEVEL_DESC[chosen_level])

    summary = load_summary(chosen_level)
    target_names = list(summary["target"])
    target_options = [(SHORT.get(n, n), n) for n in target_names]
    label_to_name = {lbl: n for (lbl, n) in target_options}

    st.markdown("---")
    st.header("Target composite")
    chosen_label = st.radio(
        "Target", list(label_to_name.keys()), label_visibility="collapsed"
    )
    chosen = label_to_name[chosen_label]

    meta = load_meta(chosen_level)
    st.markdown("---")
    st.markdown(
        f"**Cohort.** {meta['n_subjects']} healthy adults, frozen 80/20 split "
        f"({meta['n_train']} train / {meta['n_test']} test)."
    )
    if meta.get("dropped_candidates"):
        st.markdown(
            "**Dropped at this level** (linear in kept targets):  \n"
            + "  \n".join([f"`{k}` — {v}" for k, v in meta["dropped_candidates"].items()])
        )


preds, _ = load_predictions(chosen_level)
ablation = load_ablation(chosen_level)
dev_corr = load_deviation_corr(chosen_level)
dev_npz, _ = load_deviation(chosen_level)
formulations = load_formulations(chosen_level)

# ── headline metrics ─────────────────────────────────────────────────────────
row = summary[summary["target"] == chosen].iloc[0]


def _get(name, default=float("nan")):
    """Fetch a column from the headline `row` with a safe fallback so the app
    still renders against older holdout_summary.csv copies that may be cached
    by Streamlit Cloud after a deploy."""
    return row[name] if name in row.index else default


have_range = "y_train_min" in row.index
have_pct = "test_mae_pct_of_range" in row.index
n_cols = 6 if have_range else 5
cols = st.columns(n_cols)
i = 0
if have_range:
    cols[i].metric(
        "y range (train)",
        f"[{_get('y_train_min'):.2f}, {_get('y_train_max'):.2f}]",
        f"span {_get('y_train_range'):.2f}, SD {_get('y_train_std'):.2f}",
    )
    i += 1
cols[i].metric("Train-CV MAE", f"{_get('cv_mae'):.2f}", f"± {_get('cv_mae_std'):.2f}")
i += 1
test_mae_label = (
    f"{_get('test_mae_pct_of_range'):.1f}% of range"
    if have_pct
    else f"vs mean-baseline {_get('test_mae') - _get('test_mae_baseline'):+.2f}"
)
cols[i].metric("Held-out MAE", f"{_get('test_mae'):.2f}", test_mae_label); i += 1
cols[i].metric("Held-out R²", f"{_get('test_r2'):.3f}", f"baseline R² {_get('test_r2_baseline'):+.3f}"); i += 1
cols[i].metric("Held-out ρ", f"{_get('test_pearson'):.3f}"); i += 1
cols[i].metric("n_test with y", f"{int(_get('n_test_with_y'))}/86")

tabs = st.tabs([
    "Predicted vs true",
    "Per-target overview",
    "Composite formulation",
    "Modality ablation",
    "Deviation correlations",
    "Compare levels",
    "About",
])

# ── tab 1: predicted vs true ─────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Predicted vs true — {chosen_label} [{chosen_level}]")
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

# ── tab 2: per-target overview at the chosen level ───────────────────────────
with tabs[1]:
    st.subheader(f"All targets at level {chosen_level}")
    df = summary.copy()
    df["target_label"] = df["target"].map(SHORT).fillna(df["target"])
    has_range_cols = "y_train_min" in df.columns
    if has_range_cols:
        df["y_range_text"] = df.apply(
            lambda r: f"[{r['y_train_min']:.2f}, {r['y_train_max']:.2f}]  "
                      f"(span {r['y_train_range']:.2f}, SD {r['y_train_std']:.2f})",
            axis=1,
        )
        wanted = ["target_label", "n_train", "n_test_with_y", "y_range_text",
                  "cv_mae", "cv_mae_std",
                  "test_mae", "test_mae_pct_of_range", "test_rmse",
                  "test_r2", "test_pearson", "test_mae_baseline", "test_r2_baseline"]
        labels = ["Target", "n train", "n test", "y range (train)",
                  "CV MAE", "CV SD",
                  "Test MAE", "MAE / range", "Test RMSE",
                  "Test R²", "Test ρ", "Baseline MAE", "Baseline R²"]
    else:
        wanted = ["target_label", "n_train", "n_test_with_y",
                  "cv_mae", "cv_mae_std",
                  "test_mae", "test_rmse",
                  "test_r2", "test_pearson", "test_mae_baseline", "test_r2_baseline"]
        labels = ["Target", "n train", "n test",
                  "CV MAE", "CV SD", "Test MAE", "Test RMSE",
                  "Test R²", "Test ρ", "Baseline MAE", "Baseline R²"]
    df_disp = df[wanted].copy()
    float_cols = [c for c in df_disp.columns if df_disp[c].dtype.kind == "f"]
    for c in float_cols:
        if c == "test_mae_pct_of_range":
            df_disp[c] = df_disp[c].map(lambda v: f"{v:.1f}%")
        else:
            df_disp[c] = df_disp[c].map(lambda v: f"{v:.3f}")
    df_disp.columns = labels
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

# ── tab 3: composite formulation ─────────────────────────────────────────────
with tabs[2]:
    st.subheader(f"Composite formulation — [{chosen_level}]")
    if not formulations:
        st.info(
            f"No composites at the {chosen_level} level — every target is an "
            "individual subtest or measure, with no publisher-defined decomposition."
        )
    else:
        st.caption(
            "Linear regression of each composite target on its publisher-defined "
            "component subtests. **The components are not model inputs**, only "
            "documentation of internal structure. SCALED motor composites recover at "
            "R² ≈ 1.000 (linear weighting); NORM indices recover at lower R² because "
            "of the publisher's non-linear lookup-table step."
        )
        rows = []
        for tname, spec in formulations.items():
            rows.append({
                "Target": SHORT.get(tname, tname),
                "Components": ", ".join(spec.get("predictors", [])),
                "OLS R²": f"{spec.get('r2', float('nan')):.3f}",
                "n": spec.get("n", "-"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        chosen_spec = formulations.get(chosen)
        if chosen_spec is None:
            st.info(
                f"**{SHORT.get(chosen, chosen)}** is not a composite at this level "
                "— it is an individual subtest / measure. Pick a composite target "
                "from the sidebar to see its OLS coefficients."
            )
        else:
            st.markdown(f"#### Coefficients for **{SHORT.get(chosen, chosen)}**")
            coef_rows = [{"term": p, "coefficient": c}
                         for p, c in chosen_spec["coefficients"].items()]
            cdf = pd.DataFrame(coef_rows)
            fig = px.bar(
                cdf, x="coefficient", y="term", orientation="h",
                text=cdf["coefficient"].map(lambda v: f"{v:+.3f}"),
                color="coefficient", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                labels={"coefficient": "OLS coefficient", "term": ""},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=max(220, 30 * len(cdf) + 100),
                coloraxis_showscale=False, margin=dict(l=10, r=10),
            )
            st.plotly_chart(fig, width="stretch")
            st.code(chosen_spec["equation"], language=None)
            st.caption(
                f"Intercept = {chosen_spec['intercept']:+.2f}. "
                f"Fit on n = {chosen_spec['n']} complete cases. "
                f"R² = {chosen_spec['r2']:.3f}."
            )

# ── tab 4: modality ablation ─────────────────────────────────────────────────
with tabs[3]:
    st.subheader(f"Modality block ablation — {chosen_label} [{chosen_level}]")
    st.caption(
        "Δ MAE relative to the full 7-block stack. Positive bars = removing that "
        "block hurts the fit (the block carried independent signal)."
    )
    a = ablation[ablation["target"] == chosen].copy().sort_values("delta_test_mae")
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

    with st.expander(f"Full {chosen_level} ablation table (all targets × blocks)"):
        st.dataframe(ablation, hide_index=True, width="stretch")

# ── tab 5: deviation correlations ────────────────────────────────────────────
with tabs[4]:
    st.subheader(f"Directional deviation correlations (pred − actual) — {chosen_label} [{chosen_level}]")
    st.caption(
        "Per-subject signed residual `deviation = pred − actual` correlated against "
        "the 30 BASIC_Q_* questionnaire columns and the unused _ST_ measures at the "
        "same score level. Significance flag uses Benjamini-Hochberg FDR across the "
        "features tested per target.\n\n"
        "⚠ The raw residual carries regression-to-mean — the slope of deviation vs. "
        "true score is typically negative (low scorers over-predicted, high scorers "
        "under-predicted). The slope_vs_true column in the deviation_summary indicates "
        "how strong this is per target; values near −1 mean the model explains almost "
        "no variance in y."
    )

    z = dev_npz[dev_npz["target"] == chosen].dropna(subset=["deviation", "y_true"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            z, x="y_true", y="deviation", color="split", symbol="split",
            hover_data={"fold": True, "row": True, "split": False},
            labels={"y_true": "True composite score", "deviation": "Directional residual (pred − actual)"},
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
            z, x="deviation", color="split", nbins=40, opacity=0.7,
            color_discrete_map={
                "Train CV OOF (n=341)": "#7aa6c2",
                "Held-out test (n=86)": "#d2766b",
            },
            labels={"deviation": "Directional residual (pred − actual)"},
        )
        fig2.update_layout(height=420, margin=dict(l=10, r=10))
        st.plotly_chart(fig2, width="stretch")

    corr = dev_corr[dev_corr["target"] == chosen].copy()
    corr["sig"] = corr["q_fdr"] < 0.05
    pick = st.radio("Filter", ["FDR-significant", "All"], horizontal=True)
    filt = corr[corr["sig"]] if pick == "FDR-significant" else corr
    cat_pick = st.multiselect(
        "Category", sorted(corr["category"].unique()),
        default=sorted(corr["category"].unique()),
    )
    filt = filt[filt["category"].isin(cat_pick)].sort_values("pearson_r")

    fig3 = px.bar(
        filt, x="pearson_r", y="feature", orientation="h",
        color="category",
        text=filt["pearson_r"].map(lambda v: f"{v:+.2f}"),
        labels={"pearson_r": "Pearson r vs Directional residual (pred − actual)", "feature": ""},
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

# ── tab 6: cross-level comparison ────────────────────────────────────────────
with tabs[5]:
    st.subheader("Cross-level comparison")
    st.caption(
        "Held-out R² of every target at every level, side-by-side. Targets shown at a "
        "level for which they exist; LANGUAGE has SUM at RAW & SCALED and VCI at NORM; "
        "MOTOR composites only exist at SCALED."
    )
    rows = []
    for lvl in LEVELS:
        s = load_summary(lvl)
        for _, r in s.iterrows():
            rows.append({
                "level": lvl,
                "target": SHORT.get(r["target"], r["target"]),
                "target_raw": r["target"],
                "test_r2": r["test_r2"],
                "test_mae": r["test_mae"],
                "test_pearson": r["test_pearson"],
                "test_mae_baseline": r["test_mae_baseline"],
            })
    df_all = pd.DataFrame(rows)

    fig = px.bar(
        df_all, x="test_r2", y="target", color="level", barmode="group",
        orientation="h",
        text=df_all["test_r2"].map(lambda v: f"{v:.2f}"),
        labels={"test_r2": "Held-out R²", "target": ""},
        color_discrete_map={"RAW": "#8aab8d", "SCALED": "#a47bb5", "NORM": "#7aa6c2"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=520, margin=dict(l=10, r=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Side-by-side table**")
    pivot = df_all.pivot_table(index="target", columns="level", values="test_r2", aggfunc="first")
    pivot = pivot.reindex(columns=LEVELS).round(3)
    st.dataframe(pivot, width="stretch")

# ── tab 7: about ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("About this demo")
    st.markdown(
        """
        **Cohort.** 427 healthy adults (age 19–80) from the TCNL Quanta multimodal
        ageing study. Phenotype includes behavioural batteries, resting EEG, T1-w
        MRI morphometry, DTI ROI (JHU-ICBM), resting-state fMRI Schaefer-400
        functional connectivity, and pretrained deep-feature scalars (SFCN +
        Pyment).

        **Targets.** Composite scores read verbatim from the cleaned Quanta
        phenotype CSV. The model is fit independently at three score levels —
        RAW, SCALED, NORM — because the publisher's transformations between
        them are non-linear and the levels carry slightly different
        signal-to-noise. The "Composite formulation" tab reverse-engineers each
        target from its component subtests so the internal structure is
        explicit. At every level, composites that are exact linear combinations
        of the others kept at that level are dropped (e.g. `MEMORY_*_ImmMem`).

        **Model.** Per-block RidgeCV base learners produce out-of-fold
        predictions on the 341 train subjects and direct predictions on the 86
        held-out subjects across **seven blocks: BEH · EEG · MRI · DTI · DL ·
        FC · SEX**. A RidgeCV meta-learner stacks the seven OOF columns into
        the final score. Each target is fit independently. No `_ST_` or `_Q_`
        feature is ever an input. **Chronological age is intentionally excluded
        from the features** — the composite targets are already publisher-age-
        normed or age-scaled.

        **Split.** Inherited verbatim from the brain-age held-out repo: 80/20
        age × sex stratified, frozen ahead of the final run. Train-fitted
        means impute missing features. rs-fMRI FC is reduced to 50 PCs fit on
        the training subjects only.
        """
    )
    st.markdown(
        "**Source:** `tcnl2021-lab/quanta-st-composites` (public). "
        "Working repo: internal `Quanta_st_composites`. "
        "No raw subject features ship — only de-identified position-indexed "
        "prediction vectors, per-level metrics, ablation, formulations, and "
        "deviation correlations."
    )

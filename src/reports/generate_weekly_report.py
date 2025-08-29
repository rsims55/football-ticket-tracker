#!/usr/bin/env python3
"""
Generate a weekly model report:
- Best predictors (feature importances from the fitted model)
- Accuracy over the past 7 days
- Table of predicted vs actual with errors
- Suggestions / next variables to explore
- Saves markdown report and a CSV excerpt for the past week
- Optionally emails the report if reports/send_email.py is available

NEW:
- Permutation importance (sklearn.inspection.permutation_importance) on recent data
- PDPs (Partial Dependence Plots) for the top perm-important features
- SHAP summary + per-feature dependence plots (tree-based estimators supported)
  * All advanced diagnostics are optional and fail gracefully if inputs/libs are missing.

Run:
  python src/reports/generate_weekly_report.py
"""

import os
from datetime import datetime, timedelta
import warnings
import pandas as pd
import joblib
import numpy as np

# -----------------------
# Paths / Config
# -----------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/ticket_price_model.pkl")

# You moved these under data/predicted/
EVAL_LOG_PATH = os.getenv("EVAL_LOG_PATH", "data/predicted/evaluation_metrics.csv")
MERGED_OUTPUT = os.getenv("MERGED_OUTPUT", "data/predicted/merged_eval_results.csv")

REPORT_DIR = os.getenv("REPORT_DIR", "reports")
WEEKLY_DIR = os.path.join(REPORT_DIR, "weekly")
WEEK_WINDOW_DAYS = int(os.getenv("WEEK_WINDOW_DAYS", "7"))

# Optional email recipient
REPORT_RECIPIENT = os.getenv("WEEKLY_REPORT_EMAIL", "")

# Advanced diagnostics controls
ENABLE_ADV_DIAGNOSTICS = os.getenv("ENABLE_ADV_DIAGNOSTICS", "1") == "1"
# Where to try loading feature-level data for permutation/PDP/SHAP:
# Default tries merged_eval_results (often has many features used at predict time).
PERM_SOURCE_PATH = os.getenv("PERM_SOURCE_PATH", MERGED_OUTPUT)
# How many rows to sample for diagnostics (speed control)
PERM_SAMPLE_N = int(os.getenv("PERM_SAMPLE_N", "2000"))
# Number of permutation repeats
PERM_N_REPEATS = int(os.getenv("PERM_N_REPEATS", "5"))
# Top features (by permutation importance) to visualize via PDP & SHAP
TOP_FEATURES_FOR_PLOTS = int(os.getenv("TOP_FEATURES_FOR_PLOTS", "6"))
# Image format
IMG_FMT = os.getenv("REPORT_IMG_FMT", "png")  # png|svg


# -----------------------
# Helpers (existing)
# -----------------------
def _load_eval_df() -> pd.DataFrame:
    """
    Prefer the merged evaluation file if it exists, else fall back to the raw log.
    Expected columns include at least:
      startDateEastern, homeTeam, awayTeam, predicted_lowest_price, actual_lowest_price,
      abs_error, percent_error
    """
    path = MERGED_OUTPUT if os.path.exists(MERGED_OUTPUT) else EVAL_LOG_PATH
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Normalize date
    if "startDateEastern" in df.columns:
        df["startDateEastern"] = pd.to_datetime(df["startDateEastern"], errors="coerce").dt.date
    else:
        # If missing, try a common alt name
        for alt in ("start_date", "game_date", "date_est_only"):
            if alt in df.columns:
                df["startDateEastern"] = pd.to_datetime(df[alt], errors="coerce").dt.date
                break

    # Coerce numeric columns if they exist
    for col in ("predicted_lowest_price", "actual_lowest_price", "abs_error", "percent_error"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_recent_evaluations(window_days: int = WEEK_WINDOW_DAYS) -> pd.DataFrame:
    df = _load_eval_df()
    if df.empty or "startDateEastern" not in df.columns:
        return pd.DataFrame()

    cutoff = datetime.now().date() - timedelta(days=window_days)
    recent = df[df["startDateEastern"] >= cutoff].copy()

    # Sort for a nicer table (largest misses first)
    if "abs_error" in recent.columns:
        recent.sort_values(by="abs_error", ascending=False, inplace=True)

    return recent


def _format_currency(x) -> str:
    try:
        return f"${float(x):.2f}"
    except Exception:
        return ""


def _humanize_feature(name: str, importance: float) -> str:
    # Strip transformers like "num__" or "cat__"
    if "__" in name:
        prefix, base = name.split("__", 1)
    else:
        prefix, base = "", name

    if prefix == "num":
        return f"- {base.replace('_',' ')} was important, contributing {importance:.1%} to predictions."
    elif prefix == "cat":
        # split conference features nicely
        if "Conference_" in base:
            col, val = base.split("_", 1)
            return f"- Teams from the {val} {col.replace('Conference','conference').lower()} mattered, contributing {importance:.1%}."
        else:
            return f"- {base.replace('_',' ')} category influenced predictions (~{importance:.1%})."
    else:
        return f"- {base.replace('_',' ')} influenced predictions (~{importance:.1%})."


# -----------------------
# New: model & feature plumbing
# -----------------------
def _unwrap_model(model):
    """
    Returns (pipeline, preprocessor, estimator). Any can be None if absent.
    - pipeline: sklearn Pipeline if model is a Pipeline else None
    - preprocessor: ColumnTransformer (or object with get_feature_names_out) if found
    - estimator: final estimator (e.g., RandomForestRegressor)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    pipeline = model if isinstance(model, Pipeline) else None

    preprocessor = None
    estimator = model
    if pipeline is not None:
        estimator = pipeline.steps[-1][1]
        # find the first step that looks like a preprocessor
        for _, step in pipeline.steps:
            if hasattr(step, "get_feature_names_out"):
                preprocessor = step
                break
            if isinstance(step, ColumnTransformer):
                preprocessor = step

    return pipeline, preprocessor, estimator


def _expanded_feature_names(preprocessor, estimator, importances_len=None):
    """
    Best-effort to get expanded/transformed feature names (after encoder).
    """
    names = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = None

    if names is None:
        if hasattr(estimator, "feature_names_in_"):
            names = estimator.feature_names_in_
        elif importances_len is not None:
            names = np.array([f"feature_{i}" for i in range(importances_len)])
        else:
            names = None
    return np.asarray(names) if names is not None else None


def _original_feature_names(preprocessor, estimator):
    """
    Try to reconstruct the ORIGINAL model input column names expected by the pipeline.
    - For ColumnTransformer, collect 'cols' from transformers.
    - Else fall back to estimator.feature_names_in_ if present.
    """
    from sklearn.compose import ColumnTransformer
    orig = []

    if preprocessor is not None and isinstance(preprocessor, ColumnTransformer):
        try:
            for name, trans, cols in preprocessor.transformers_:
                if cols == "drop":
                    continue
                if cols == "remainder":
                    # can't easily enumerate remainder here; skip
                    continue
                if isinstance(cols, (list, tuple, np.ndarray)):
                    for c in cols:
                        if isinstance(c, str):
                            orig.append(c)
                # If it's a callable/slice, we can't resolve—skip
        except Exception:
            pass

    if not orig and hasattr(estimator, "feature_names_in_"):
        # For bare estimator trained directly on original columns
        orig = list(estimator.feature_names_in_)

    # Deduplicate, preserve order
    seen = set()
    ordered = []
    for c in orig:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _coerce_booleans_inplace(X: pd.DataFrame, cols):
    """
    Lightly coerce boolean-ish columns to {0,1} or proper bools.
    """
    true_set = {"true", "1", "yes", "y", "t"}
    false_set = {"false", "0", "no", "n", "f"}
    for c in cols:
        if c in X.columns:
            s = X[c]
            if pd.api.types.is_bool_dtype(s):
                continue
            if pd.api.types.is_numeric_dtype(s):
                continue
            # try mapping strings
            try:
                X[c] = s.map(lambda v: np.nan if pd.isna(v) else str(v).strip().lower())
                X[c] = X[c].map(lambda v: True if v in true_set else False if v in false_set else v)
                # If still strings, leave as-is and let pipeline handle OHE
            except Exception:
                pass


def _load_perm_dataset(orig_feature_list, target_col="actual_lowest_price"):
    """
    Load a DataFrame to drive permutation importance & PDP/SHAP.
    - Prefer PERM_SOURCE_PATH. If missing, try MERGED_OUTPUT then EVAL_LOG_PATH.
    - Keep only rows that have the target if available.
    - Return (X_df, y_series, raw_df)
    """
    candidates = [PERM_SOURCE_PATH, MERGED_OUTPUT, EVAL_LOG_PATH]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if path is None:
        return None, None, None

    df = pd.read_csv(path)

    # If target not present, try common alternatives
    if target_col not in df.columns:
        for alt in ("actual_price", "actual", "y", "lowest_price", "y_true"):
            if alt in df.columns:
                target_col = alt
                break

    # Filter to rows with target (if present)
    y = df[target_col].astype(float) if target_col in df.columns else None

    # Build X with only original feature columns we actually have
    keep_cols = [c for c in orig_feature_list if c in df.columns]
    if not keep_cols:
        return None, y, df

    X = df[keep_cols].copy()

    # Light coercions for booleans that pipelines typically expect
    boolish = [c for c in keep_cols if c.lower().startswith("is") or c.lower().endswith("flag") or "flag" in c.lower()]
    _coerce_booleans_inplace(X, boolish)

    # Drop rows with NaNs in target if we have y
    if y is not None:
        m = pd.notna(y)
        X, y = X[m], y[m]

    # Sample for speed
    if len(X) > PERM_SAMPLE_N:
        X = X.sample(PERM_SAMPLE_N, random_state=42)
        if y is not None:
            y = y.loc[X.index]

    return X, y, df


# -----------------------
# Existing: feature_importance (from model)
# -----------------------
def get_feature_importance(top_k: int = 20) -> tuple[str, list[str]]:
    """
    Returns (markdown_text, weak_features_list).
    Handles:
      - Pipeline(ColumnTransformer(...), RandomForestRegressor)
      - Bare RandomForestRegressor with feature_names_in_
    Produces:
      - Expanded top-K transformed features
      - Aggregated importances by original input column (sum over one-hot levels)
    """
    if not os.path.exists(MODEL_PATH):
        return "❌ Model file not found.", []

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return f"❌ Failed to load model: {e}", []

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = None
    estimator = model
    feature_names_expanded = None

    if isinstance(model, Pipeline):
        estimator = getattr(model, "steps", [(-1, model)])[-1][1]
        for _, step in model.steps:
            if hasattr(step, "get_feature_names_out"):
                preprocessor = step
                break
            if isinstance(step, ColumnTransformer):
                preprocessor = step

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return "❌ Model does not expose feature_importances_.", []

    importances = np.asarray(importances)

    # expanded feature names
    feature_names_expanded = _expanded_feature_names(preprocessor, estimator, importances_len=len(importances))
    if feature_names_expanded is None:
        feature_names_expanded = np.array([f"feature_{i}" for i in range(len(importances))])

    # Length guard
    n = min(len(feature_names_expanded), len(importances))
    feature_names_expanded = np.asarray(feature_names_expanded[:n], dtype=str)
    importances = importances[:n]

    # expanded (top-K) markdown
    order = np.argsort(importances)[::-1]
    top_idx = order[:top_k]
    lines_expanded = [
        _humanize_feature(feature_names_expanded[i], importances[i])
        for i in top_idx
    ]

    # aggregate by original column
    base_map = {}
    for name, imp in zip(feature_names_expanded, importances):
        base = name.split("__", 1)[-1]
        if "_" in base:
            base = base.rsplit("_", 1)[0]
        base_map[base] = base_map.get(base, 0.0) + float(imp)

    agg_items = sorted(base_map.items(), key=lambda x: x[1], reverse=True)
    lines_agg = [f"- {k}: {v:.4f}" for k, v in agg_items[:top_k]]

    weak_features = [k for k, v in agg_items if v < 0.01]

    md = []
    md.append("### Top Transformed Features (expanded)")
    md.extend(lines_expanded if lines_expanded else ["(none)"])
    md.append("\n### Aggregated by Original Column")
    md.extend(lines_agg if lines_agg else ["(none)"])
    if weak_features:
        md.append("\n**Possibly unrelated (near-zero importance):** " + ", ".join(weak_features[:20]))

    return "\n".join(md), weak_features


def _safe_rmse(df: pd.DataFrame) -> float:
    """
    Compute RMSE from predicted vs actual when available; otherwise approximate
    from abs_error if that's all we have (less ideal).
    """
    if {"predicted_lowest_price", "actual_lowest_price"}.issubset(df.columns):
        diff2 = (df["predicted_lowest_price"] - df["actual_lowest_price"]) ** 2
        return float(np.sqrt(np.nanmean(diff2)))
    if "abs_error" in df.columns:
        return float(np.sqrt(np.nanmean((df["abs_error"]) ** 2)))
    return float("nan")


# -----------------------
# New: Permutation Importance
# -----------------------
def run_permutation_importance(model, X, y):
    """
    Runs permutation importance using the given model (Pipeline or Estimator).
    Returns a DataFrame with columns: feature, mean_importance, std_importance
    """
    from sklearn.inspection import permutation_importance

    if X is None or X.empty:
        return None

    if y is None or y.isna().all():
        # can fall back to model.score with synthetic y? Better to bail out.
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X, y,
            n_repeats=PERM_N_REPEATS,
            random_state=42,
            n_jobs=-1
        )

    # If X is a DataFrame, use its columns as the "original" feature names
    feat_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(result.importances_mean.shape[0])]
    df = pd.DataFrame({
        "feature": feat_names,
        "mean_importance": result.importances_mean,
        "std_importance": result.importances_std
    }).sort_values("mean_importance", ascending=False)
    return df


# -----------------------
# New: PDP generation
# -----------------------
def _sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def generate_pdp_plots(model, X, feature_names, out_dir, prefix="pdp"):
    """
    Saves PDP plots for the provided original feature names (using the Pipeline if possible).
    Returns list of image paths that were created.
    """
    created = []
    if X is None or X.empty or not feature_names:
        return created

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        from sklearn.inspection import PartialDependenceDisplay
    except Exception as e:
        print(f"⚠️  Skipping PDP (matplotlib/sklearn not available): {e}")
        return created

    # Only keep features present in X
    feats = [f for f in feature_names if f in X.columns]
    for f in feats:
        try:
            fig = plt.figure(figsize=(6, 4))
            ax = plt.gca()
            # For pipelines, sklearn will handle preprocessing internally
            PartialDependenceDisplay.from_estimator(model, X, [f], ax=ax)
            ax.set_title(f"PDP: {f}")
            out = os.path.join(out_dir, f"{prefix}_{_sanitize_filename(f)}.{IMG_FMT}")
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            created.append(out)
        except Exception as e:
            print(f"⚠️  PDP failed for {f}: {e}")
            continue

    return created


# -----------------------
# New: SHAP diagnostics
# -----------------------
def _map_expanded_to_original(expanded_names):
    """
    Build a mapping {original_base_col: [expanded_indices]} using the same heuristic as before:
    - Strip transformer prefix up to first "__"
    - Then rsplit "_" once to collapse one-hot category to base column
    """
    base_to_idx = {}
    for i, name in enumerate(expanded_names):
        base = name.split("__", 1)[-1]
        if "_" in base:
            base = base.rsplit("_", 1)[0]
        base_to_idx.setdefault(base, []).append(i)
    return base_to_idx


def run_shap_and_plots(estimator, preprocessor, X_orig, top_original_feats, out_dir, prefix="shap"):
    """
    Compute SHAP for tree-based estimators on the transformed design matrix.
    - Aggregates mean |SHAP| by original column (summing over one-hot).
    - Saves a summary bar chart and per-feature dependence plots (choosing the
      highest-impact expanded column as representative).
    Returns (agg_table_path, summary_plot_path, per_feature_paths)
    """
    if X_orig is None or X_orig.empty:
        return None, None, []

    try:
        import shap  # type: ignore
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"⚠️  Skipping SHAP (library not available): {e}")
        return None, None, []

    # Only tree-based explainers supported here
    tree_like = hasattr(estimator, "predict") and any(k in estimator.__class__.__name__.lower() for k in ("forest", "tree", "xgb", "lgbm", "catboost", "gbm", "gradientboost"))
    if not tree_like:
        print("ℹ️  SHAP: estimator is not tree-based; skipping.")
        return None, None, []

    # Transform X -> expanded design
    try:
        X_trans = preprocessor.transform(X_orig) if preprocessor is not None else X_orig.values
    except Exception as e:
        print(f"⚠️  SHAP: failed to transform X with preprocessor: {e}")
        return None, None, []

    # Ensure dense
    try:
        import scipy
        if scipy.sparse.issparse(X_trans):
            X_trans = X_trans.toarray()
    except Exception:
        # if scipy not present, try .toarray if available
        X_trans = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans)

    expanded_names = _expanded_feature_names(preprocessor, estimator, importances_len=X_trans.shape[1])
    if expanded_names is None or len(expanded_names) != X_trans.shape[1]:
        expanded_names = np.array([f"feature_{i}" for i in range(X_trans.shape[1])])

    # Tree explainer
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans)
        # For regressors, shap_values is (n_samples, n_features)
        if isinstance(shap_values, list):
            # occasional multi-output shape—pick the first
            shap_values = shap_values[0]
    except Exception as e:
        print(f"⚠️  SHAP computation failed: {e}")
        return None, None, []

    # Aggregate mean |SHAP| by original feature
    base_to_idx = _map_expanded_to_original(expanded_names)
    rows = []
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    for base, idxs in base_to_idx.items():
        rows.append((base, float(np.sum(mean_abs[idxs]))))
    agg = pd.DataFrame(rows, columns=["feature", "mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False)

    # Save aggregated table
    agg_path = os.path.join(out_dir, f"{prefix}_mean_abs_by_feature_{datetime.now().date()}.{IMG_FMT.replace(IMG_FMT, 'csv')}")
    # (quick trick to ensure .csv)
    agg_path = agg_path.replace(f".{IMG_FMT}", ".csv")
    agg.to_csv(agg_path, index=False)

    # Summary bar (top 20)
    try:
        import matplotlib.pyplot as plt
        top20 = agg.head(20).iloc[::-1]
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.barh(top20["feature"], top20["mean_abs_shap"])
        ax.set_title("SHAP Mean |Impact| by Feature (Top 20)")
        ax.set_xlabel("Mean |SHAP|")
        fig.tight_layout()
        summary_plot_path = os.path.join(out_dir, f"{prefix}_summary_bar.{IMG_FMT}")
        fig.savefig(summary_plot_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"⚠️  SHAP summary bar failed: {e}")
        summary_plot_path = None

    # Per-feature dependence plots for the chosen original features:
    per_feature_paths = []
    for base in top_original_feats:
        idxs = base_to_idx.get(base, [])
        if not idxs:
            continue
        # pick the expanded col with highest mean_abs
        rep_idx = int(sorted(idxs, key=lambda i: mean_abs[i], reverse=True)[0])
        try:
            fig = plt.figure(figsize=(6, 4))
            shap.dependence_plot(
                rep_idx, shap_values, X_trans,
                feature_names=list(expanded_names), show=False
            )
            out = os.path.join(out_dir, f"{prefix}_dependence_{_sanitize_filename(base)}.{IMG_FMT}")
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            per_feature_paths.append(out)
        except Exception as e:
            print(f"⚠️  SHAP dependence plot failed for {base}: {e}")

    return agg_path, summary_plot_path, per_feature_paths


# -----------------------
# Report builder (existing + new sections)
# -----------------------
def build_report() -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(WEEKLY_DIR, exist_ok=True)

    report_md_path = os.path.join(WEEKLY_DIR, f"weekly_report_{today_str}.md")
    recent_csv_path = os.path.join(WEEKLY_DIR, f"weekly_eval_rows_{today_str}.csv")

    report = [f"# 📈 Weekly Ticket Price Model Report\n**Date:** {today_str}\n"]

    # -----------------------
    # Section 1: Feature Importance (from fitted model)
    # -----------------------
    report.append("## 🔍 Best Predictors of Ticket Price\n")
    fi_text, weak_features = get_feature_importance()
    report.append(fi_text + "\n")

    # -----------------------
    # Section 1b: Permutation Importance (NEW)
    # -----------------------
    perm_csv_path = None
    pdp_imgs, shap_summary_img, shap_dep_imgs, shap_table_path = [], None, [], None

    if ENABLE_ADV_DIAGNOSTICS and os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            pipeline, preprocessor, estimator = _unwrap_model(model)
            model_for_perm = model  # Use the pipeline directly if available

            # Find original feature columns expected
            orig_feats = _original_feature_names(preprocessor, estimator)
            X_perm, y_perm, _raw = _load_perm_dataset(orig_feats)

            perm_df = run_permutation_importance(model_for_perm, X_perm, y_perm)
            if perm_df is not None and not perm_df.empty:
                # Save CSV
                perm_csv_path = os.path.join(WEEKLY_DIR, f"permutation_importance_{today_str}.csv")
                perm_df.to_csv(perm_csv_path, index=False)

                # Write section
                report.append("## 🧪 Permutation Importance (recent data)\n")
                topN = perm_df.head(20)
                report.append("Top features by mean importance:\n")
                for _, r in topN.iterrows():
                    report.append(f"- {r['feature']}: {r['mean_importance']:.6f} (±{r['std_importance']:.6f})")
                report.append("")
                report.append(f"_Saved full table → `{os.path.relpath(perm_csv_path, start='.')}`_\n")

                # Determine features for PDP/SHAP
                top_for_plots = [f for f in perm_df.head(TOP_FEATURES_FOR_PLOTS)["feature"].tolist() if f in (X_perm.columns if X_perm is not None else [])]

                # PDPs
                if top_for_plots:
                    pdp_imgs = generate_pdp_plots(model_for_perm, X_perm, top_for_plots, WEEKLY_DIR, prefix=f"pdp_{today_str}")
                    if pdp_imgs:
                        report.append("## 📈 Partial Dependence (Top Perm-Important)\n")
                        for img in pdp_imgs:
                            rel = os.path.relpath(img, start='.')
                            report.append(f"![PDP]({rel})")
                        report.append("")

                # SHAP
                # Use estimator (tree-based) + transformed design
                agg_path, shap_summary_img, shap_dep_imgs = None, None, []
                try:
                    agg_path, shap_summary_img, shap_dep_imgs = run_shap_and_plots(
                        estimator=estimator,
                        preprocessor=preprocessor,
                        X_orig=X_perm,
                        top_original_feats=top_for_plots,
                        out_dir=WEEKLY_DIR,
                        prefix=f"shap_{today_str}"
                    )
                except Exception as e:
                    print(f"⚠️  SHAP diagnostics failed: {e}")

                if agg_path or shap_summary_img or shap_dep_imgs:
                    report.append("## 🧮 SHAP Diagnostics (Top Perm-Important)\n")
                    if agg_path and os.path.exists(agg_path):
                        report.append(f"- Aggregated mean |SHAP| table: `{os.path.relpath(agg_path, start='.')}`")
                    if shap_summary_img and os.path.exists(shap_summary_img):
                        report.append(f"![SHAP Summary]({os.path.relpath(shap_summary_img, start='.')})")
                    for img in shap_dep_imgs:
                        report.append(f"![SHAP Dependence]({os.path.relpath(img, start='.')})")
                    report.append("")

        except Exception as e:
            report.append(f"### ⚠️ Advanced diagnostics skipped\nReason: {e}\n")

    # -----------------------
    # Section 2: Accuracy (Past Week)  (existing)
    # -----------------------
    df = get_recent_evaluations(WEEK_WINDOW_DAYS)
    if df.empty:
        report.append("## 📊 Model Accuracy (Past 7 Days)\nNo games to evaluate in the past week.\n")
    else:
        # Persist the slice for easier inspection later
        cols_to_save = [
            c for c in [
                "startDateEastern", "homeTeam", "awayTeam",
                "predicted_lowest_price", "actual_lowest_price",
                "abs_error", "percent_error", "weekNumber",
                "dayOfWeek", "kickoffHour"
            ] if c in df.columns
        ]
        df[cols_to_save].to_csv(recent_csv_path, index=False)

        report.append("## 📊 Model Accuracy (Past 7 Days)\n")
        report.append(f"- Games evaluated: **{len(df)}**")

        mae = float(df["abs_error"].mean()) if "abs_error" in df.columns else float("nan")
        rmse = _safe_rmse(df)
        over_5 = int((df["percent_error"] > 0.05).sum()) if "percent_error" in df.columns else 0

        if not np.isnan(mae):
            report.append(f"- MAE: **{_format_currency(mae)}**")
        if not np.isnan(rmse):
            report.append(f"- RMSE: **{_format_currency(rmse)}**")
        if "percent_error" in df.columns:
            report.append(f"- Games > 5% error: **{over_5} / {len(df)}**\n")

        # -----------------------
        # Section 3: Table of predictions (existing)
        # -----------------------
        report.append("## 🎯 Predicted vs Actual Prices\n")
        report.append("| Game | Date (ET) | Predicted | Actual | Abs Error | % Error |")
        report.append("|------|-----------|-----------|--------|-----------|---------|")

        for _, row in df.iterrows():
            game = f"{row.get('homeTeam','')} vs {row.get('awayTeam','')}"
            date_str = row.get("startDateEastern", "")
            p = row.get("predicted_lowest_price", float("nan"))
            a = row.get("actual_lowest_price", float("nan"))
            ae = row.get("abs_error", float("nan"))
            pe = row.get("percent_error", float("nan"))
            pe_pct = f"{pe * 100:.1f}%" if pd.notna(pe) else ""
            report.append(
                f"| {game} | {date_str} | {_format_currency(p)} | {_format_currency(a)} | {_format_currency(ae)} | {pe_pct} |"
            )

        # -----------------------
        # Section 4: Heuristic suggestions (existing)
        # -----------------------
        report.append("\n## 💡 Suggestions")
        if "percent_error" in df.columns and len(df) > 0 and (over_5 / len(df)) > 0.40:
            report.append("- Miss rate >40% this week; consider revisiting hyperparameters or adding interaction features.")
        if any(col in df.columns for col in ["homeTeamRank", "awayTeamRank"]):
            if "homeTeamRank" in df.columns and df["homeTeamRank"].isna().any():
                report.append("- Some home rankings are missing; verify postseason/final AP pulls.")
            if "awayTeamRank" in df.columns and df["awayTeamRank"].isna().any():
                report.append("- Some away rankings are missing; verify postseason/final AP pulls.")
        report.append("- Consider adding: team momentum (last 2–3 games), previous-week result diff, rivalry strength score, and weather (temp/precip).")
        report.append("- Explore time-of-day effects more granularly (hour buckets) and weekday/weekend splits.")
        report.append("- Check stadium capacity normalization (capacity vs. sold % if/when available).\n")

        if weak_features:
            report.append(
                "- Near-zero importance this week (may be unrelated): "
                + ", ".join(sorted(set(weak_features))[:20])
            )

    # -----------------------
    # Write report file
    # -----------------------
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"✅ Weekly report saved to {report_md_path}")
    if not df.empty and os.path.exists(recent_csv_path):
        print(f"🗂  Weekly eval rows saved to {recent_csv_path}")
    if perm_csv_path and os.path.exists(perm_csv_path):
        print(f"🧪 Permutation importance saved to {perm_csv_path}")

    # -----------------------
    # Optional email hook
    # -----------------------
    try:
        if REPORT_RECIPIENT:
            from reports.send_email import send_markdown_report  # your existing helper
            send_markdown_report(report_md_path, REPORT_RECIPIENT)
            print(f"📧 Report emailed to {REPORT_RECIPIENT}")
    except Exception as e:
        print(f"⚠️  Skipping email send (not configured or failed): {e}")

    return report_md_path


if __name__ == "__main__":
    build_report()

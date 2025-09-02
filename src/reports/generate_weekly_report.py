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
- Permutation importance on recent data
- PDPs for top perm-important features
- SHAP summary + dependence plots
- Robust model loader that monkey-patches missing private sklearn symbols
- Auto-completion of missing original columns for diagnostics (neutral defaults)
- Per-run dated folder with subfolders:
    reports/weekly/YYYY-MM-DD/
      weekly_report_YYYY-MM-DD.md
      images/  (all plots here)
      data/    (all CSVs here)

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
PERM_SOURCE_PATH = os.getenv("PERM_SOURCE_PATH", MERGED_OUTPUT)
PERM_SAMPLE_N = int(os.getenv("PERM_SAMPLE_N", "2000"))
PERM_N_REPEATS = int(os.getenv("PERM_N_REPEATS", "5"))
TOP_FEATURES_FOR_PLOTS = int(os.getenv("TOP_FEATURES_FOR_PLOTS", "6"))
IMG_FMT = os.getenv("REPORT_IMG_FMT", "png")  # png|svg

TZ_LABEL = "ET"  # display-only label for times


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility: monkey-patch for scikit-learn pickle changes
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_sklearn_unpickle_compat():
    try:
        import sklearn
        print(f"[weekly_report] scikit-learn version: {sklearn.__version__}")
        from sklearn.compose import _column_transformer as ct
        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                """Minimal stand-in for deprecated private sklearn class."""
                pass
            ct._RemainderColsList = _RemainderColsList
    except Exception as e:
        # Non-fatal; loading may still work
        print(f"[weekly_report] compat patch warning: {e}")


def _robust_load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    _ensure_sklearn_unpickle_compat()
    return joblib.load(path)


# -----------------------
# Helpers
# -----------------------
def _compose_game_label(row: pd.Series) -> str:
    # Try a bunch of common field names; fall back to event_id
    home_keys = ["homeTeam", "home_team", "home", "homeTeamName", "homeSchool", "home_name"]
    away_keys = ["awayTeam", "away_team", "away", "awayTeamName", "awaySchool", "away_name"]

    def _first(keys):
        for k in keys:
            v = row.get(k, "")
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        return ""

    home = _first(home_keys)
    away = _first(away_keys)

    if home or away:
        return f"{home} vs {away}".strip(" vs ")
    # fallback: conferences, then event_id
    hc = str(row.get("homeConference", "") or "").strip()
    ac = str(row.get("awayConference", "") or "").strip()
    if hc or ac:
        return f"{hc} vs {ac}".strip(" vs ")
    eid = row.get("event_id", "")
    return f"event {eid}" if pd.notna(eid) and str(eid) else ""


def _sort_for_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows for table rendering: highest absolute $ error first, then keep others."""
    if "abs_error" in df.columns:
        # numeric coerce just in case
        s = pd.to_numeric(df["abs_error"], errors="coerce")
        return df.assign(_abs_error_num=s).sort_values("_abs_error_num", ascending=False, na_position="last").drop(columns=["_abs_error_num"])
    return df


def _md_rel(from_dir: str, path: str) -> str:
    p = os.path.relpath(path, start=from_dir)
    return p.replace("\\", "/")


def _parse_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _load_eval_df() -> pd.DataFrame:
    """
    Prefer MERGED_OUTPUT, else EVAL_LOG_PATH.
    Expected (from evaluator):
      startDateEastern, predicted_lowest_price, actual_lowest_price, abs_error, percent_error
      # === NEW: TIMING ===
      predicted_optimal_dt, actual_lowest_dt, timing_abs_error_hours, timing_signed_error_hours
    """
    path = MERGED_OUTPUT if os.path.exists(MERGED_OUTPUT) else EVAL_LOG_PATH
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Keep both a date-only version (for window filter) and full datetimes where provided
    if "startDateEastern" in df.columns:
        df["_startDateEastern_dt"] = _parse_dt(df["startDateEastern"])
        df["startDateEastern"] = _parse_dt(df["startDateEastern"]).dt.date
    else:
        for alt in ("start_date", "game_date", "date_est_only"):
            if alt in df.columns:
                df["_startDateEastern_dt"] = _parse_dt(df[alt])
                df["startDateEastern"] = _parse_dt(df[alt]).dt.date
                break

    # Price columns
    for col in ("predicted_lowest_price", "actual_lowest_price", "abs_error", "percent_error"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # === NEW: TIMING ===
    if "predicted_optimal_dt" in df.columns:
        df["predicted_optimal_dt"] = _parse_dt(df["predicted_optimal_dt"])
    if "actual_lowest_dt" in df.columns:
        df["actual_lowest_dt"] = _parse_dt(df["actual_lowest_dt"])
    for col in ("timing_abs_error_hours", "timing_signed_error_hours"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_recent_evaluations(window_days: int = WEEK_WINDOW_DAYS) -> pd.DataFrame:
    df = _load_eval_df()
    if df.empty or "startDateEastern" not in df.columns:
        return pd.DataFrame()

    cutoff = datetime.now().date() - timedelta(days=window_days)
    recent = df[df["startDateEastern"] >= cutoff].copy()

    # Sort by largest price misses first (stable key if present)
    if "abs_error" in recent.columns:
        recent.sort_values(by="abs_error", ascending=False, inplace=True)

    return recent


def _format_currency(x) -> str:
    try:
        return f"${float(x):.2f}"
    except Exception:
        return ""


def _format_dt(dt) -> str:
    if pd.isna(dt):
        return ""
    return pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M")


def _humanize_feature(name: str, importance: float) -> str:
    if "__" in name:
        prefix, base = name.split("__", 1)
    else:
        prefix, base = "", name

    if prefix == "num":
        return f"- {base.replace('_',' ')} was important, contributing {importance:.1%} to predictions."
    elif prefix == "cat":
        if "Conference_" in base:
            col, val = base.split("_", 1)
            return f"- Teams from the {val} {col.replace('Conference','conference').lower()} mattered, contributing {importance:.1%}."
        else:
            return f"- {base.replace('_',' ')} category influenced predictions (~{importance:.1%})."
    else:
        return f"- {base.replace('_',' ')} influenced predictions (~{importance:.1%})."


# -----------------------
# Model & feature plumbing (unchanged)
# -----------------------
def _unwrap_model(model):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    pipeline = model if isinstance(model, Pipeline) else None

    preprocessor = None
    estimator = model
    if pipeline is not None:
        estimator = pipeline.steps[-1][1]
        for _, step in pipeline.steps:
            if hasattr(step, "get_feature_names_out"):
                preprocessor = step
                break
            if isinstance(step, ColumnTransformer):
                preprocessor = step

    return pipeline, preprocessor, estimator


def _expanded_feature_names(preprocessor, estimator, importances_len=None):
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
    from sklearn.compose import ColumnTransformer
    orig = []

    if preprocessor is not None and isinstance(preprocessor, ColumnTransformer):
        try:
            for name, trans, cols in preprocessor.transformers_:
                if cols == "drop":
                    continue
                if cols == "remainder":
                    continue
                if isinstance(cols, (list, tuple, np.ndarray)):
                    for c in cols:
                        if isinstance(c, str):
                            orig.append(c)
        except Exception:
            pass

    if not orig and hasattr(estimator, "feature_names_in_"):
        orig = list(estimator.feature_names_in_)

    seen = set()
    ordered = []
    for c in orig:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def _coerce_booleans_inplace(X: pd.DataFrame, cols):
    true_set = {"true", "1", "yes", "y", "t"}
    false_set = {"false", "0", "no", "n", "f"}
    for c in cols:
        if c in X.columns:
            s = X[c]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                continue
            try:
                X[c] = s.map(lambda v: np.nan if pd.isna(v) else str(v).strip().lower())
                X[c] = X[c].map(lambda v: True if v in true_set else False if v in false_set else v)
            except Exception:
                pass


def _guess_kind(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["days_until", "rank", "capacity", "week", "hour", "listing", "count"]):
        return "num"
    if n.startswith("num__") or n.startswith("num_"):
        return "num"
    if n.startswith("is") or n.endswith("flag") or "flag" in n or n in {"neutralsite", "conferencegame"}:
        return "bool"
    if n.startswith("cat__") or n.startswith("cat_"):
        return "cat"
    return "cat"


def _complete_required_columns(X: pd.DataFrame,
                               required_cols: list[str],
                               preprocessor=None) -> pd.DataFrame:
    Xc = X.copy()
    missing = [c for c in required_cols if c not in Xc.columns]
    for c in missing:
        kind = _guess_kind(c)
        if kind == "bool":
            Xc[c] = False
        elif kind == "num":
            Xc[c] = 0.0
        else:
            Xc[c] = "__MISSING__"
    cols = [c for c in required_cols] + [c for c in Xc.columns if c not in required_cols]
    return Xc[cols]


def _load_perm_dataset(orig_feature_list, target_col="actual_lowest_price"):
    candidates = [
        os.getenv("PERM_SOURCE_PATH", ""),
        MERGED_OUTPUT,
        EVAL_LOG_PATH,
        "data/predicted/predicted_prices_optimal.csv",
        "data/daily/price_snapshots.csv",
        "data/enriched/schedule_enriched.csv",
    ]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if path is None:
        return None, None, None

    df = pd.read_csv(path)

    if target_col not in df.columns:
        for alt in ("actual_lowest_price", "actual_price", "actual", "y", "lowest_price", "y_true"):
            if alt in df.columns:
                target_col = alt
                break

    y = df[target_col].astype(float) if target_col in df.columns else None

    keep_cols = [c for c in orig_feature_list if c in df.columns]
    X_raw = df[keep_cols].copy() if keep_cols else pd.DataFrame(index=df.index)

    boolish = [c for c in keep_cols if c.lower().startswith("is") or c.lower().endswith("flag") or "flag" in c.lower()]
    _coerce_booleans_inplace(X_raw, boolish)

    X = _complete_required_columns(X_raw, orig_feature_list)

    if y is not None:
        m = pd.notna(y)
        X, y = X[m], y[m]

    if len(X) > PERM_SAMPLE_N:
        X = X.sample(PERM_SAMPLE_N, random_state=42)
        if y is not None:
            y = y.loc[X.index]

    return X, y, df


# -----------------------
# Existing: feature_importance (from model)
# -----------------------
def get_feature_importance(top_k: int = 20) -> tuple[str, list[str]]:
    if not os.path.exists(MODEL_PATH):
        return "❌ Model file not found.", []

    try:
        model = _robust_load_model(MODEL_PATH)
    except Exception as e:
        return f"❌ Failed to load model: {e}", []

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    preprocessor = None
    estimator = model

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

    feature_names_expanded = _expanded_feature_names(preprocessor, estimator, importances_len=len(importances))
    if feature_names_expanded is None:
        feature_names_expanded = np.array([f"feature_{i}" for i in range(len(importances))])

    n = min(len(feature_names_expanded), len(importances))
    feature_names_expanded = np.asarray(feature_names_expanded[:n], dtype=str)
    importances = importances[:n]

    order = np.argsort(importances)[::-1]
    top_idx = order[:top_k]
    lines_expanded = [
        _humanize_feature(feature_names_expanded[i], importances[i])
        for i in top_idx
    ]

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
    if {"predicted_lowest_price", "actual_lowest_price"}.issubset(df.columns):
        diff2 = (df["predicted_lowest_price"] - df["actual_lowest_price"]) ** 2
        return float(np.sqrt(np.nanmean(diff2)))
    if "abs_error" in df.columns:
        return float(np.sqrt(np.nanmean((df["abs_error"]) ** 2)))
    return float("nan")


# -----------------------
# Permutation Importance
# -----------------------
def run_permutation_importance(model, X, y):
    from sklearn.inspection import permutation_importance

    if X is None or X.empty:
        return None

    if y is None or y.isna().all():
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X, y,
            n_repeats=PERM_N_REPEATS,
            random_state=42,
            n_jobs=-1
        )

    feat_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(result.importances_mean.shape[0])]
    df = pd.DataFrame({
        "feature": feat_names,
        "mean_importance": result.importances_mean,
        "std_importance": result.importances_std
    }).sort_values("mean_importance", ascending=False)
    return df


# -----------------------
# PDP generation
# -----------------------
def _sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def generate_pdp_plots(model, X, feature_names, out_dir, prefix="pdp"):
    created = []
    if X is None or X.empty or not feature_names:
        return created

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.inspection import PartialDependenceDisplay
    except Exception as e:
        print(f"⚠️  Skipping PDP (matplotlib/sklearn not available): {e}")
        return created

    feats = [f for f in feature_names if f in X.columns]
    for f in feats:
        try:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.gca()
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
# SHAP diagnostics
# -----------------------
def _map_expanded_to_original(expanded_names):
    base_to_idx = {}
    for i, name in enumerate(expanded_names):
        base = name.split("__", 1)[-1]
        if "_" in base:
            base = base.rsplit("_", 1)[0]
        base_to_idx.setdefault(base, []).append(i)
    return base_to_idx


def run_shap_and_plots(estimator, preprocessor, X_orig, top_original_feats, out_dir, prefix="shap"):
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

    tree_like = hasattr(estimator, "predict") and any(k in estimator.__class__.__name__.lower() for k in ("forest", "tree", "xgb", "lgbm", "catboost", "gbm", "gradientboost"))
    if not tree_like:
        print("ℹ️  SHAP: estimator is not tree-based; skipping.")
        return None, None, []

    try:
        X_trans = preprocessor.transform(X_orig) if preprocessor is not None else X_orig.values
    except Exception as e:
        print(f"⚠️  SHAP: failed to transform X with preprocessor: {e}")
        return None, None, []

    try:
        import scipy
        if scipy.sparse.issparse(X_trans):
            X_trans = X_trans.toarray()
    except Exception:
        X_trans = X_trans.toarray() if hasattr(X_trans, "toarray") else np.asarray(X_trans)

    expanded_names = _expanded_feature_names(preprocessor, estimator, importances_len=X_trans.shape[1])
    if expanded_names is None or len(expanded_names) != X_trans.shape[1]:
        expanded_names = np.array([f"feature_{i}" for i in range(X_trans.shape[1])])

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception as e:
        print(f"⚠️  SHAP computation failed: {e}")
        return None, None, []

    base_to_idx = _map_expanded_to_original(expanded_names)
    rows = []
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    for base, idxs in base_to_idx.items():
        rows.append((base, float(np.sum(mean_abs[idxs]))))
    agg = pd.DataFrame(rows, columns=["feature", "mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False)

    agg_path = os.path.join(os.path.dirname(out_dir), "data", f"{prefix}_mean_abs_by_feature_{datetime.now().date()}.csv")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    agg.to_csv(agg_path, index=False)

    try:
        import matplotlib.pyplot as plt
        top20 = agg.head(20).iloc[::-1]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
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

    per_feature_paths = []
    for base in top_original_feats:
        idxs = base_to_idx.get(base, [])
        if not idxs:
            continue
        rep_idx = int(sorted(idxs, key=lambda i: mean_abs[i], reverse=True)[0])
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 4))
            import shap as _shap  # reuse
            _shap.dependence_plot(
                rep_idx, shap_values, X_trans,
                feature_names=list(expanded_names), show=False
            )
            fig = plt.gcf()
            out = os.path.join(out_dir, f"{prefix}_dependence_{_sanitize_filename(base)}.{IMG_FMT}")
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            per_feature_paths.append(out)
        except Exception as e:
            print(f"⚠️  SHAP dependence plot failed for {base}: {e}")

    return agg_path, summary_plot_path, per_feature_paths


# -----------------------
# Report builder
# -----------------------
def build_report() -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")

    report_dir_for_date = os.path.join(WEEKLY_DIR, today_str)
    images_dir = os.path.join(report_dir_for_date, "images")
    data_dir = os.path.join(report_dir_for_date, "data")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    report_md_path = os.path.join(report_dir_for_date, f"weekly_report_{today_str}.md")
    recent_csv_path = os.path.join(data_dir, f"weekly_eval_rows_{today_str}.csv")

    report = [f"# 📈 Weekly Ticket Price Model Report\n**Date:** {today_str}\n"]

    # 1) Feature Importance
    report.append("## 🔍 Best Predictors of Ticket Price\n")
    fi_text, weak_features = get_feature_importance()
    report.append(fi_text + "\n")

    # 1b) Permutation / PDP / SHAP (unchanged)
    perm_csv_path = None
    if ENABLE_ADV_DIAGNOSTICS and os.path.exists(MODEL_PATH):
        try:
            model = _robust_load_model(MODEL_PATH)
            pipeline, preprocessor, estimator = _unwrap_model(model)
            model_for_perm = model
            orig_feats = _original_feature_names(preprocessor, estimator)
            if not orig_feats:
                raise RuntimeError("Could not recover original feature names from the model/pipeline.")
            X_perm, y_perm, _raw = _load_perm_dataset(orig_feats)
            if X_perm is None:
                raise RuntimeError("No diagnostics dataset found (PERM_SOURCE_PATH/MERGED_OUTPUT/EVAL_LOG_PATH missing).")
            missing_cols = [c for c in orig_feats if c not in X_perm.columns]
            if missing_cols:
                raise RuntimeError(f"columns are missing even after completion: {set(missing_cols)}")
            perm_df = run_permutation_importance(model_for_perm, X_perm, y_perm)
            if perm_df is not None and not perm_df.empty:
                perm_csv_path = os.path.join(data_dir, f"permutation_importance_{today_str}.csv")
                perm_df.to_csv(perm_csv_path, index=False)
                report.append("## 🧪 Permutation Importance (recent data)\n")
                topN = perm_df.head(20)
                report.append("Top features by mean importance:\n")
                for _, r in topN.iterrows():
                    report.append(f"- {r['feature']}: {r['mean_importance']:.6f} (±{r['std_importance']:.6f})")
                report.append("")
                report.append(f"_Saved full table → `{_md_rel(report_dir_for_date, perm_csv_path)}`_\n")
                top_for_plots = [f for f in perm_df.head(TOP_FEATURES_FOR_PLOTS)["feature"].tolist() if f in (X_perm.columns if X_perm is not None else [])]
                if top_for_plots:
                    pdp_imgs = generate_pdp_plots(model_for_perm, X_perm, top_for_plots, images_dir, prefix=f"pdp_{today_str}")
                    if pdp_imgs:
                        report.append("## 📈 Partial Dependence (Top Perm-Important)\n")
                        for img in pdp_imgs:
                            report.append(f"![PDP]({_md_rel(report_dir_for_date, img)})")
                        report.append("")
                agg_path, shap_summary_img, shap_dep_imgs = None, None, []
                try:
                    agg_path, shap_summary_img, shap_dep_imgs = run_shap_and_plots(
                        estimator=estimator,
                        preprocessor=preprocessor,
                        X_orig=X_perm,
                        top_original_feats=top_for_plots,
                        out_dir=images_dir,
                        prefix=f"shap_{today_str}"
                    )
                except Exception as e:
                    print(f"⚠️  SHAP diagnostics failed: {e}")
                if agg_path or shap_summary_img or shap_dep_imgs:
                    report.append("## 🧮 SHAP Diagnostics (Top Perm-Important)\n")
                    if agg_path and os.path.exists(agg_path):
                        report.append(f"- Aggregated mean |SHAP| table: `{_md_rel(report_dir_for_date, agg_path)}`")
                    if shap_summary_img and os.path.exists(shap_summary_img):
                        report.append(f"![SHAP Summary]({_md_rel(report_dir_for_date, shap_summary_img)})")
                    for img in shap_dep_imgs:
                        report.append(f"![SHAP Dependence]({_md_rel(report_dir_for_date, img)})")
                    report.append("")
        except Exception as e:
            report.append(f"### ⚠️ Advanced diagnostics skipped\nReason: {e}\n")

    # 2) Accuracy (past week) + TIMING
    df = get_recent_evaluations(WEEK_WINDOW_DAYS)
    if df.empty:
        report.append("## 📊 Model Accuracy (Past 7 Days)\nNo games to evaluate in the past week.\n")
    else:
        # Save weekly slice (including timing columns if present)
        cols_to_save = [
            c for c in [
                "startDateEastern", "homeTeam", "awayTeam",
                "predicted_lowest_price", "actual_lowest_price",
                "abs_error", "percent_error", "weekNumber",
                "dayOfWeek", "kickoffHour",
                # === NEW: TIMING ===
                "predicted_optimal_dt", "actual_lowest_dt",
                "timing_abs_error_hours", "timing_signed_error_hours",
            ] if c in df.columns
        ]
        df[cols_to_save].to_csv(recent_csv_path, index=False)

        report.append("## 📊 Model Accuracy (Past 7 Days)\n")
        report.append(f"- Games evaluated: **{len(df)}**")

        mae = float(df["abs_error"].mean()) if "abs_error" in df.columns else float("nan")
        rmse = _safe_rmse(df)
        over_5 = int((df["percent_error"] > 0.05).sum()) if "percent_error" in df.columns else 0

        if not np.isnan(mae):
            report.append(f"- MAE (price): **{_format_currency(mae)}**")
        if not np.isnan(rmse):
            report.append(f"- RMSE (price): **{_format_currency(rmse)}**")
        if "percent_error" in df.columns:
            report.append(f"- Games > 5% price error: **{over_5} / {len(df)}**")

        # === NEW: TIMING SUMMARY ===
        if "timing_abs_error_hours" in df.columns and df["timing_abs_error_hours"].notna().any():
            t_mae = float(df["timing_abs_error_hours"].mean())
            t_med = float(df["timing_abs_error_hours"].median())
            within_6h = int((df["timing_abs_error_hours"] <= 6).sum())
            within_12h = int((df["timing_abs_error_hours"] <= 12).sum())
            within_24h = int((df["timing_abs_error_hours"] <= 24).sum())
            bias = float(df.get("timing_signed_error_hours", pd.Series(dtype=float)).mean()) if "timing_signed_error_hours" in df.columns else float("nan")

            report.append("\n### ⏱️ Timing Accuracy (Predicted Optimal vs Actual Lowest)")
            report.append(f"- MAE (hours): **{t_mae:.2f} h**  •  Median |Δ|: **{t_med:.2f} h**")
            report.append(f"- Within 6h: **{within_6h}/{len(df)}**  •  Within 12h: **{within_12h}/{len(df)}**  •  Within 24h: **{within_24h}/{len(df)}**")
            if not np.isnan(bias):
                direction = "later than" if bias > 0 else "earlier than"
                report.append(f"- Bias: predictions are on average **{abs(bias):.2f} h {direction}** actual lows")

        # 3) Table (price + timing)
        has_timing = {"predicted_optimal_dt", "actual_lowest_dt", "timing_abs_error_hours"}.issubset(df.columns)

        # Always sort the view we print: highest dollar abs error first
        def _sort_for_table(_df: pd.DataFrame) -> pd.DataFrame:
            if "abs_error" in _df.columns:
                s = pd.to_numeric(_df["abs_error"], errors="coerce")
                return _df.assign(_abs_error_num=s).sort_values("_abs_error_num", ascending=False, na_position="last").drop(columns=["_abs_error_num"])
            return _df

        table_df = _sort_for_table(df.copy())

        if has_timing:
            report.append("\n## 🎯 Predicted vs Actual Prices & Timing\n")
            report.append(f"| Game | Date ({TZ_LABEL}) | Pred $ | Actual $ | Abs $ | % Err | Pred Opt ({TZ_LABEL}) | Actual Low ({TZ_LABEL}) | Abs Δ (h) |")
            report.append("|------|--------------------|--------|----------|-------|-------|----------------------|-------------------------|-----------|")
        else:
            report.append("\n## 🎯 Predicted vs Actual Prices\n")
            report.append("| Game | Date (ET) | Predicted | Actual | Abs Error | % Error |")
            report.append("|------|-----------|-----------|--------|-----------|---------|")

        for _, row in table_df.iterrows():
            game = _compose_game_label(row)
            date_str = row.get("startDateEastern", "")
            p = row.get("predicted_lowest_price", float("nan"))
            a = row.get("actual_lowest_price", float("nan"))
            ae = row.get("abs_error", float("nan"))
            pe = row.get("percent_error", float("nan"))
            pe_pct = f"{pe * 100:.1f}%" if pd.notna(pe) else ""

            if has_timing:
                pdt = _format_dt(row.get("predicted_optimal_dt", pd.NaT))
                adt = _format_dt(row.get("actual_lowest_dt", pd.NaT))
                td = row.get("timing_abs_error_hours", np.nan)
                td_s = "" if pd.isna(td) else f"{float(td):.2f}"
                report.append(
                    f"| {game} | {date_str} | {_format_currency(p)} | {_format_currency(a)} | "
                    f"{_format_currency(ae)} | {pe_pct} | {pdt} | {adt} | {td_s} |"
                )
            else:
                report.append(
                    f"| {game} | {date_str} | {_format_currency(p)} | {_format_currency(a)} | "
                    f"{_format_currency(ae)} | {pe_pct} |"
                )


        # 4) Suggestions
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
        report.append("- Check stadium capacity normalization (capacity vs. sold % if/when available).")
        if "timing_signed_error_hours" in df.columns and df["timing_signed_error_hours"].notna().any():
            later_share = float((df["timing_signed_error_hours"] > 0).mean())
            report.append(f"- Timing: {later_share:.0%} of predictions occur *after* the actual low — consider features about pre-game demand decay and listing churn.")
        if weak_features:
            report.append("- Near-zero importance this week (may be unrelated): " + ", ".join(sorted(set(weak_features))[:20]))

    # Write report file
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"✅ Weekly report saved to {report_md_path}")
    if not df.empty and os.path.exists(recent_csv_path):
        print(f"🗂  Weekly eval rows saved to {recent_csv_path}")
    if 'perm_csv_path' in locals() and perm_csv_path and os.path.exists(perm_csv_path):
        print(f"🧪 Permutation importance saved to {perm_csv_path}")

    # Optional email hook
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

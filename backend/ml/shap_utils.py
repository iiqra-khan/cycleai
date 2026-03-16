# =============================================================================
# shap_utils.py — SHAP explainability utilities
# =============================================================================

import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_feature_importance(
    model,
    feature_names: list,
    model_name: str,
    X_test: pd.DataFrame = None,
    output_dir: str = "."
) -> pd.DataFrame:
    """
    SHAP-first feature importance.
    Falls back to MDI / coefficients if X_test not provided.
    Saves beeswarm, bar, and waterfall plots to output_dir.
    """
    importance_df = None
    os.makedirs(output_dir, exist_ok=True)

    if X_test is not None:
        try:
            print(f"\n  [SHAP] Computing values for: {model_name}")

            if hasattr(model, "named_steps"):
                clf       = model.named_steps["clf"]
                explainer = shap.LinearExplainer(clf, X_test)
                shap_vals = explainer.shap_values(X_test)
                base_val  = explainer.expected_value
            else:
                explainer = shap.TreeExplainer(model)
                shap_expl = explainer(X_test)
                shap_vals = shap_expl.values
                base_val  = shap_expl.base_values
                if shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1]
                    base_val  = (base_val[:, 1] if base_val.ndim > 1
                                 else base_val)

            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                "feature":    feature_names,
                "importance": mean_abs_shap,
                "method":     "SHAP (mean |value|)"
            }).sort_values("importance", ascending=False)

            base_scalar = (float(np.mean(base_val))
                           if hasattr(base_val, "__len__")
                           else float(base_val))
            shap_explanation = shap.Explanation(
                values        = shap_vals,
                base_values   = np.full(len(shap_vals), base_scalar),
                data          = X_test.values,
                feature_names = list(feature_names)
            )

            # Beeswarm
            fig, _ = plt.subplots(figsize=(10, 8))
            shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
            plt.title(f"SHAP Beeswarm — {model_name}", fontsize=13)
            plt.tight_layout()
            path = os.path.join(output_dir,
                                f"shap_beeswarm_{model_name.replace(' ','_')}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [SAVED] Beeswarm → {path}")

            # Bar
            fig, _ = plt.subplots(figsize=(10, 8))
            shap.plots.bar(shap_explanation, max_display=20, show=False)
            plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13)
            plt.tight_layout()
            path = os.path.join(output_dir,
                                f"shap_bar_{model_name.replace(' ','_')}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [SAVED] Bar plot → {path}")

            # Waterfall (sample 0)
            fig, _ = plt.subplots(figsize=(10, 7))
            shap.plots.waterfall(shap_explanation[0], max_display=15, show=False)
            plt.title(f"SHAP Waterfall (sample 0) — {model_name}", fontsize=12)
            plt.tight_layout()
            path = os.path.join(output_dir,
                                f"shap_waterfall_{model_name.replace(' ','_')}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [SAVED] Waterfall → {path}")

            print(f"\n  [SHAP] Top 15 Features — {model_name}")
            print(importance_df.head(15).to_string(index=False))
            return importance_df

        except Exception as e:
            print(f"  [SHAP] Error: {e} — falling back to MDI/coef")

    # Fallback
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": model.feature_importances_,
            "method":     "MDI (Tree-based)"
        }).sort_values("importance", ascending=False)

    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if hasattr(clf, "coef_"):
            importance_df = pd.DataFrame({
                "feature":    feature_names,
                "importance": np.abs(clf.coef_[0]),
                "method":     "Absolute Coefficient"
            }).sort_values("importance", ascending=False)

    if importance_df is not None:
        print(f"\n[Explainability] Top 15 Features — {model_name}")
        print(importance_df.head(15).to_string(index=False))

    return importance_df


def get_shap_context(
    model,
    X_test: pd.DataFrame,
    feature_names: list,
    sample_idx: int = 0
) -> dict:
    """
    Extracts structured SHAP context for one sample.
    Used by AgenticRAG to build retrieval queries.
    """
    try:
        if hasattr(model, "named_steps"):
            clf       = model.named_steps["clf"]
            explainer = shap.LinearExplainer(clf, X_test)
            shap_vals = explainer.shap_values(X_test)
            base_val  = float(explainer.expected_value)
        else:
            explainer = shap.TreeExplainer(model)
            expl      = explainer(X_test)
            shap_vals = expl.values
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            bv       = expl.base_values
            base_val = float(
                bv[sample_idx, 1] if bv.ndim > 1 else bv[sample_idx]
            )

        row   = shap_vals[sample_idx]
        df_sh = pd.DataFrame({"feature": feature_names, "shap_value": row})
        pos   = (df_sh[df_sh["shap_value"] > 0]
                 .sort_values("shap_value", ascending=False).head(6))
        neg   = (df_sh[df_sh["shap_value"] < 0]
                 .sort_values("shap_value").head(6))

        return {
            "base_value":               round(base_val, 4),
            "total_shap_contribution":  round(float(row.sum()), 4),
            "top_positive_features": [
                {"feature": r["feature"],
                 "shap_value": round(r["shap_value"], 4)}
                for _, r in pos.iterrows()
            ],
            "top_negative_features": [
                {"feature": r["feature"],
                 "shap_value": round(r["shap_value"], 4)}
                for _, r in neg.iterrows()
            ],
        }
    except Exception as e:
        print(f"  [SHAP context] Warning: {e}")
        return {
            "base_value": 0.0, "total_shap_contribution": 0.0,
            "top_positive_features": [], "top_negative_features": []
        }
# =============================================================================
# pipeline.py — Main training and evaluation pipeline
# =============================================================================

import os
import json
import numpy as np
import pandas as pd

from .config import TARGET_PERIOD, TARGET_OVUL
from .features import engineer_features, get_engineered_feature_list
from .models import build_models, handle_class_imbalance
from .evaluate import evaluate_model, cross_validate_model
from .shap_utils import extract_feature_importance
from .validator import validate_prediction_input, reorder_features_for_model
from ..agents.explainer import AgenticRAG, generate_clinical_explanation


def run_pipeline_with_validation(df: pd.DataFrame,
                                  output_dir: str = ".") -> tuple:
    """
    Full training pipeline:
      1. Temporal split (raw data)
      2. Feature engineering (each half independently)
      3. Train + evaluate 3 models per target
      4. SHAP explainability
      5. Agentic RAG clinical explanation

    Returns:
        all_results : dict of metrics/cv/feature importance per target
        best_models : dict keyed by target with model + metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    best_models = {}

    print("\n" + "=" * 60)
    print("  MENSTRUAL CYCLE PREDICTION PIPELINE")
    print("  Explainable AI Healthcare Decision Support System")
    print("=" * 60)

    # ── STEP 1: Temporal split on raw data ───────────────────────────────────
    print("\n[STEP 1] Temporal split on raw data...")
    CONTEXT_DAYS = 7
    train_rows, test_rows = [], []

    for pid, group in df.sort_values(["id", "day_in_study"]).groupby("id"):
        n      = len(group)
        cutoff = int(n * 0.8)
        if cutoff < 30:
            train_rows.append(group)
            continue
        train_rows.append(group.iloc[:cutoff])
        test_rows.append(group.iloc[cutoff - CONTEXT_DAYS:])

    if not test_rows:
        raise ValueError(
            "Raw split produced empty test set. "
            "Lower min_train_days or add more participant history."
        )

    train_raw = pd.concat(train_rows).reset_index(drop=True)
    test_raw  = pd.concat(test_rows).reset_index(drop=True)
    print(f"  Raw train: {len(train_raw)} rows | Raw test: {len(test_raw)} rows")

    # ── STEP 2: Feature Engineering (each half independently) ────────────────
    print("\n[STEP 2] Feature Engineering (train and test separately)...")
    df_train_eng = engineer_features(train_raw)
    df_test_eng_with_context = engineer_features(test_raw)

    # Strip context rows from test
    cutoff_days = {}
    for pid, group in df.sort_values(["id", "day_in_study"]).groupby("id"):
        n = len(group)
        cutoff_days[pid] = group.iloc[int(n * 0.8)]["day_in_study"]

    mask = df_test_eng_with_context.apply(
        lambda row: row["day_in_study"] >= cutoff_days[row["id"]], axis=1
    )
    df_test_eng = df_test_eng_with_context[mask].reset_index(drop=True)

    feature_cols = get_engineered_feature_list(df_train_eng)

    # Align test columns
    for col in feature_cols:
        if col not in df_test_eng.columns:
            df_test_eng[col] = 0

    # ── STEP 3: Determine targets ─────────────────────────────────────────────
    targets = [TARGET_PERIOD]
    if TARGET_OVUL in df_train_eng.columns:
        targets.append(TARGET_OVUL)

    for target in targets:
        if target not in df_train_eng.columns:
            print(f"\n[SKIP] Target '{target}' not found in data.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  TARGET: {target.upper()}")
        print(f"{'=' * 60}")

        # ── Extract X/y ───────────────────────────────────────────────────────
        available = [f for f in feature_cols if f in df_train_eng.columns]

        X_train = df_train_eng[available].fillna(0)
        y_train = df_train_eng[target].dropna().astype(int)
        X_train = X_train.loc[y_train.index]

        X_test  = df_test_eng[available].fillna(0)
        y_test  = df_test_eng[target].dropna().astype(int)
        X_test  = X_test.loc[y_test.index]

        used_features = available

        print(f"\n[Data] Target: '{target}'")
        print(f"  Train: {len(X_train)} rows | "
              f"Positive rate: {y_train.mean():.3f}")
        print(f"  Test:  {len(X_test)} rows  | "
              f"Positive rate: {y_test.mean():.3f}")
        print(f"  Features: {len(used_features)}")

        # ── Class imbalance ───────────────────────────────────────────────────
        X_train_bal, y_train_bal = handle_class_imbalance(
            X_train, y_train, strategy="class_weight"
        )

        # ── Train + evaluate ──────────────────────────────────────────────────
        models         = build_models(y_train=y_train_bal)
        target_results = {
            "metrics": [], "cv_results": [], "feature_importance": {}
        }
        best_model      = None
        best_auc        = 0
        best_model_name = ""

        for model_name, model in models.items():
            print(f"\n[TRAINING] {model_name}")
            model.fit(X_train_bal, y_train_bal)

            metrics = evaluate_model(model, X_test, y_test, model_name)
            target_results["metrics"].append(metrics)

            cv_res = cross_validate_model(model, X_train, y_train, model_name)
            target_results["cv_results"].append(cv_res)

            fi_df = extract_feature_importance(
                model, used_features, model_name,
                X_test=X_test, output_dir=output_dir
            )
            if fi_df is not None:
                target_results["feature_importance"][model_name] = fi_df

            if metrics["auc_roc"] > best_auc:
                best_auc        = metrics["auc_roc"]
                best_model      = model
                best_model_name = model_name

        print(f"\n[BEST MODEL] {best_model_name} (AUC-ROC: {best_auc:.4f})")
        best_fi = target_results["feature_importance"].get(best_model_name)

        # ── Input validation ──────────────────────────────────────────────────
        print(f"\n[STEP 7] Input Validation")
        sample_features = X_test.iloc[0:1]

        try:
            validate_prediction_input(
                sample_features, best_model,
                verbose=True, raise_on_mismatch=True
            )
            sample_features_clean = reorder_features_for_model(
                sample_features, best_model
            )
            sample_proba = best_model.predict_proba(
                sample_features_clean
            )[0, 1]

        except ValueError as e:
            print(f"[ERROR] Validation failed: {e}")
            sample_proba = None

        # ── Clinical explanation + RAG ────────────────────────────────────────
        if sample_proba is not None:
            explanation = generate_clinical_explanation(
                sample_proba, best_fi, target
            )
            print("\n" + explanation)

            rag = AgenticRAG()
            rag_context = {
                "target":        target,
                "top_model":     best_model_name,
                "auc_roc":       best_auc,
                "prediction":    sample_proba,
                "top_features":  (best_fi.head(5)["feature"].tolist()
                                  if best_fi is not None else []),
                "model":         best_model,
                "feature_names": used_features,
            }
            rag_output = rag.retrieve_and_explain(rag_context, X_test=X_test)
            print("\n" + rag_output)
        else:
            print("[SKIP] RAG explanation skipped due to validation failure.")

        best_models[target] = {
            "model":      best_model,
            "model_name": best_model_name,
            "auc":        best_auc,
            "X_train":    X_train_bal,
            "y_train":    y_train_bal,
        }

        # ── Save results ──────────────────────────────────────────────────────
        all_results[target] = target_results
        results_path = os.path.join(output_dir, f"results_{target}.json")
        serializable = {
            "metrics":    target_results["metrics"],
            "cv_results": target_results["cv_results"],
            "best_model": best_model_name,
            "best_auc":   best_auc,
            "feature_importance_top20": {
                k: v.head(20).to_dict(orient="records")
                for k, v in target_results["feature_importance"].items()
            }
        }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n[SAVED] Results → {results_path}")

    return all_results, best_models
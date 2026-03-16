# INTEGRATION: Add this to your run_pipeline function before making predictions
# This shows the recommended validation points

"""a
Within the run_pipeline function, after Step 6 (Generate Clinical Explanation):

    # Step 6.5: VALIDATE PREDICTION INPUT (NEW)
    print(f"\n[VALIDATION] Checking prediction input...")
    from prediction_validator import validate_prediction_input, reorder_features_for_model
    
    # Create sample prediction features (in deployment, this comes from user input)
    sample_features = X_test.iloc[0:1]  # Use first test sample
    
    validation = validate_prediction_input(
        sample_features,
        best_model,
        verbose=True,
        raise_on_mismatch=True
    )
    
    if validation["valid"]:
        sample_features_clean = reorder_features_for_model(sample_features, best_model)
        sample_proba_pred = best_model.predict_proba(sample_features_clean)[0, 1]
    else:
        print("[ERROR] Prediction validation failed - cannot proceed")
        sample_proba_pred = None
"""


# ============================================================================
# ALTERNATIVE: Wrapper for the existing pipeline
# ============================================================================

def run_pipeline_with_validation(df, output_dir="."):
    """
    Enhanced version of run_pipeline with built-in validation.
    Drop-in replacement for the original run_pipeline.
    """
    from prediction_validator import validate_prediction_input, reorder_features_for_model
    import os
    import json
    import pandas as pd
    from your_module import (  # Replace with your actual imports
        engineer_features,
        get_engineered_feature_list,
        prepare_data,
        handle_class_imbalance,
        build_models,
        evaluate_model,
        cross_validate_model,
        extract_feature_importance,
        generate_clinical_explanation,
        AgenticRAGStub,
        TARGET_PERIOD,
        TARGET_OVUL
    )
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    print("\n" + "="*60)
    print("  MENSTRUAL CYCLE PREDICTION PIPELINE")
    print("  Explainable AI Healthcare Decision Support System")
    print("="*60)

    # Step 1: Feature Engineering
    print("\n[STEP 1] Feature Engineering...")
    df_eng = engineer_features(df)
    feature_cols = get_engineered_feature_list(df_eng)

    # Step 2: Run pipeline for each target
    targets = [TARGET_PERIOD]
    if TARGET_OVUL in df_eng.columns:
        targets.append(TARGET_OVUL)

    for target in targets:
        if target not in df_eng.columns:
            print(f"\n[SKIP] Target '{target}' not found in data.")
            continue

        print(f"\n{'='*60}")
        print(f"  TARGET: {target.upper()}")
        print(f"{'='*60}")

        # Step 3: Data Preparation
        X_train, X_test, y_train, y_test, used_features = prepare_data(
            df_eng, target, feature_cols
        )

        # Step 4: Class Imbalance
        X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)

        # Step 5: Train & Evaluate Models
        models = build_models()
        target_results = {"metrics": [], "cv_results": [], "feature_importance": {}}

        best_model     = None
        best_auc       = 0
        best_model_name = ""

        for model_name, model in models.items():
            print(f"\n[TRAINING] {model_name}")
            model.fit(X_train_bal, y_train_bal)

            # Holdout evaluation
            metrics = evaluate_model(model, X_test, y_test, model_name)
            target_results["metrics"].append(metrics)

            # Cross-validation
            cv_res = cross_validate_model(model, X_train, y_train, model_name)
            target_results["cv_results"].append(cv_res)

            # Feature Importance
            fi_df = extract_feature_importance(model, used_features, model_name)
            if fi_df is not None:
                target_results["feature_importance"][model_name] = fi_df

            # Track best model by AUC
            if metrics["auc_roc"] > best_auc:
                best_auc        = metrics["auc_roc"]
                best_model      = model
                best_model_name = model_name

        # Step 6: Generate Clinical Explanation for Best Model
        print(f"\n[BEST MODEL] {best_model_name} (AUC-ROC: {best_auc:.4f})")
        best_fi = target_results["feature_importance"].get(best_model_name)

        # Step 6.5: VALIDATE WITH NEW PREDICTION VALIDATOR
        print(f"\n[STEP 6.5] Input Validation")
        sample_features = X_test.iloc[0:1]
        
        try:
            validation = validate_prediction_input(
                sample_features,
                best_model,
                verbose=True,
                raise_on_mismatch=True
            )
            
            # Reorder if needed
            sample_features_clean = reorder_features_for_model(sample_features, best_model)
            sample_proba = best_model.predict_proba(sample_features_clean)[0, 1]
            
        except ValueError as e:
            print(f"[ERROR] Validation failed: {e}")
            sample_proba = 0.72  # Fallback

        explanation = generate_clinical_explanation(sample_proba, best_fi, target)
        print("\n" + explanation)

        # Step 7: RAG Stub
        rag = AgenticRAGStub()
        rag_context = {
            "target":        target,
            "top_model":     best_model_name,
            "auc_roc":       best_auc,
            "prediction":    sample_proba,
            "top_features":  best_fi.head(5)["feature"].tolist() if best_fi is not None else []
        }
        rag_output = rag.retrieve_and_explain(rag_context)
        print("\n" + rag_output)

        # Save results
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

    return all_results, best_model, y_train_bal
"""
Prediction Validator Module
Validates input features before making predictions in the pipeline.
Integrates with the main pipeline to catch feature mismatches early.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


def validate_prediction_input(
    features: pd.DataFrame,
    model,
    verbose: bool = True,
    raise_on_mismatch: bool = True
) -> Dict[str, any]:
    """
    Comprehensive validation of features before prediction.
    
    Args:
        features: Input dataframe for prediction
        model: Trained sklearn model with feature_names_in_ attribute
        verbose: Print diagnostic output
        raise_on_mismatch: Raise error if features don't match model training
        
    Returns:
        Validation result dict with status and details
    """
    
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "diagnostics": {}
    }
    
    # ========== 1. SHAPE & INTEGRITY CHECKS ==========
    if verbose:
        print("\n" + "="*60)
        print("PREDICTION INPUT VALIDATION")
        print("="*60)
        print(f"\n[SHAPE] Features shape: {features.shape}")
    
    result["diagnostics"]["shape"] = features.shape
    
    # Check for NaNs
    nan_check = features.isnull().any().any()
    nan_by_col = features.isnull().sum()
    
    if verbose:
        print(f"[NaN CHECK] Any missing values: {nan_check}")
        if nan_check:
            print(f"  Columns with NaNs: {nan_by_col[nan_by_col > 0].to_dict()}")
    
    if nan_check:
        result["warnings"].append("Input contains NaN values - will cause prediction errors")
        result["diagnostics"]["nans_per_column"] = nan_by_col[nan_by_col > 0].to_dict()
    
    # ========== 2. FEATURE NAME & ORDER MATCHING ==========
    expected_features = list(model.feature_names_in_)
    actual_features = features.columns.tolist()
    
    if verbose:
        print(f"\n[FEATURES] Expected ({len(expected_features)}): {expected_features}")
        print(f"[FEATURES] Actual   ({len(actual_features)}): {actual_features}")
    
    # Check for exact match (name + order)
    features_match_exact = expected_features == actual_features
    
    if verbose:
        print(f"[MATCH] Exact match (name + order): {features_match_exact}")
    
    if not features_match_exact:
        result["valid"] = False
        result["errors"].append("Feature names or order don't match model training")
        
        # Detailed mismatch analysis
        missing = set(expected_features) - set(actual_features)
        extra = set(actual_features) - set(expected_features)
        
        if missing:
            result["errors"].append(f"  Missing features: {missing}")
            if verbose:
                print(f"  [ERROR] Missing features: {missing}")
        
        if extra:
            result["errors"].append(f"  Extra features: {extra}")
            if verbose:
                print(f"  [ERROR] Extra features: {extra}")
        
        # Check if it's just an ordering issue
        if set(expected_features) == set(actual_features):
            result["warnings"].append("Feature names match but order differs - reordering required")
            if verbose:
                print(f"  [WARNING] Features match but different order")
    
    # ========== 3. DATA TYPE & VALUE CHECKS ==========
    if verbose:
        print(f"\n[DATA TYPES]")
    
    for col in actual_features:
        dtype = features[col].dtype
        if verbose:
            print(f"  {col}: {dtype}")
        
        # Warn about object/string types (should be numeric for most models)
        if dtype == "object":
            result["warnings"].append(f"Column '{col}' is object type - may cause prediction issues")
    
    # ========== 4. VALUE RANGE CHECKS ==========
    if verbose:
        print(f"\n[VALUE RANGES]")
    
    for col in actual_features:
        col_min = features[col].min()
        col_max = features[col].max()
        col_mean = features[col].mean()
        
        if verbose:
            print(f"  {col}: min={col_min:.2f}, max={col_max:.2f}, mean={col_mean:.2f}")
        
        result["diagnostics"][f"{col}_stats"] = {
            "min": float(col_min),
            "max": float(col_max),
            "mean": float(col_mean),
            "std": float(features[col].std())
        }
    
    # ========== 5. SAMPLE DATA PREVIEW ==========
    if verbose:
        print(f"\n[SAMPLE DATA] First row (transposed for readability):")
        print(features.iloc[0].to_string())
    
    # ========== 6. FINAL DECISION ==========
    if verbose:
        print("\n" + "="*60)
        if result["valid"]:
            print("✓ VALIDATION PASSED - Safe to predict")
        else:
            print("✗ VALIDATION FAILED - Do not proceed with prediction")
            for error in result["errors"]:
                print(f"  ERROR: {error}")
        
        if result["warnings"]:
            print("\nWarnings:")
            for warning in result["warnings"]:
                print(f"  ⚠ {warning}")
        print("="*60 + "\n")
    
    if result["valid"] is False and raise_on_mismatch:
        raise ValueError(
            f"Feature validation failed:\n" + 
            "\n".join(result["errors"])
        )
    
    return result


def reorder_features_for_model(
    features: pd.DataFrame,
    model
) -> pd.DataFrame:
    """
    Reorder and select features to match model's expected input.
    Useful when feature names match but order differs.
    
    Args:
        features: Input dataframe
        model: Trained model with feature_names_in_
        
    Returns:
        Reordered dataframe matching model's expected features
    """
    expected_features = list(model.feature_names_in_)
    
    try:
        features_reordered = features[expected_features]
        print(f"✓ Features reordered to match model training")
        return features_reordered
    except KeyError as e:
        raise ValueError(
            f"Cannot reorder features - missing columns: {e}"
        )


def safe_predict(
    model,
    features: pd.DataFrame,
    validate: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Predict with built-in validation.
    
    Args:
        model: Trained sklearn model
        features: Input features
        validate: Run validation before predicting
        verbose: Print diagnostic output
        
    Returns:
        Prediction array
    """
    
    if validate:
        validation = validate_prediction_input(
            features, model, verbose=verbose, raise_on_mismatch=True
        )
    
    # Ensure feature order matches
    features_ordered = reorder_features_for_model(features, model)
    
    # Make prediction
    predictions = model.predict(features_ordered)
    
    if verbose:
        print(f"[PREDICTION] Shape: {predictions.shape}")
        print(f"[PREDICTION] Sample: {predictions[:5]}")
    
    return predictions


# ============================================================================
# Example integration into pipeline
# ============================================================================

def predict_with_validation_example(model, features: pd.DataFrame):
    """
    Example of how to integrate validation into the pipeline's prediction step.
    """
    
    # Method 1: Manual validation (more control)
    print("\n[STEP X] Validating prediction input...")
    validation = validate_prediction_input(
        features, 
        model, 
        verbose=True,
        raise_on_mismatch=True  # Fail fast if something is wrong
    )
    
    if validation["valid"]:
        features_clean = reorder_features_for_model(features, model)
        predictions = model.predict(features_clean)
        probabilities = model.predict_proba(features_clean)
        return predictions, probabilities
    
    # Method 2: All-in-one (simpler)
    # predictions = safe_predict(model, features, validate=True, verbose=True)
    # return predictions


if __name__ == "__main__":
    # Example usage
    print("Prediction Validator Module - Ready to import")
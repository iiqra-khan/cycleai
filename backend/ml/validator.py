# =============================================================================
# validator.py — Prediction input validation
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict


def validate_prediction_input(
    features: pd.DataFrame,
    model,
    verbose: bool = True,
    raise_on_mismatch: bool = True
) -> Dict:
    """
    Comprehensive validation of features before prediction.
    Returns a result dict with status and details.
    """
    result = {"valid": True, "errors": [], "warnings": [], "diagnostics": {}}

    if verbose:
        print("\n" + "=" * 60)
        print("PREDICTION INPUT VALIDATION")
        print("=" * 60)
        print(f"\n[SHAPE] Features shape: {features.shape}")

    result["diagnostics"]["shape"] = features.shape

    # NaN check
    nan_check  = features.isnull().any().any()
    nan_by_col = features.isnull().sum()
    if verbose:
        print(f"[NaN CHECK] Any missing values: {nan_check}")
    if nan_check:
        result["warnings"].append("Input contains NaN values")
        result["diagnostics"]["nans_per_column"] = nan_by_col[nan_by_col > 0].to_dict()

    # Feature name + order match
    expected = list(model.feature_names_in_)
    actual   = features.columns.tolist()
    exact    = expected == actual

    if verbose:
        print(f"\n[FEATURES] Expected ({len(expected)}): {expected}")
        print(f"[FEATURES] Actual   ({len(actual)}): {actual}")
        print(f"[MATCH] Exact match (name + order): {exact}")

    if not exact:
        result["valid"] = False
        missing = set(expected) - set(actual)
        extra   = set(actual)   - set(expected)
        if missing:
            result["errors"].append(f"Missing features: {missing}")
        if extra:
            result["errors"].append(f"Extra features: {extra}")
        if set(expected) == set(actual):
            result["warnings"].append("Features match but order differs — reorder required")

    # Data types
    if verbose:
        print(f"\n[DATA TYPES]")
        for col in actual:
            print(f"  {col}: {features[col].dtype}")

    # Value ranges
    if verbose:
        print(f"\n[VALUE RANGES]")
        for col in actual:
            print(f"  {col}: min={features[col].min():.2f}, "
                  f"max={features[col].max():.2f}, "
                  f"mean={features[col].mean():.2f}")

    if verbose:
        print(f"\n[SAMPLE DATA] First row (transposed for readability):")
        print(features.iloc[0].to_string())
        print("\n" + "=" * 60)
        if result["valid"]:
            print("✓ VALIDATION PASSED - Safe to predict")
        else:
            print("✗ VALIDATION FAILED - Do not proceed with prediction")
            for err in result["errors"]:
                print(f"  ERROR: {err}")
        print("=" * 60 + "\n")

    if not result["valid"] and raise_on_mismatch:
        raise ValueError(
            "Feature validation failed:\n" + "\n".join(result["errors"])
        )

    return result


def reorder_features_for_model(features: pd.DataFrame, model) -> pd.DataFrame:
    """Reorder features to match model's expected input."""
    expected = list(model.feature_names_in_)
    try:
        reordered = features[expected]
        print("✓ Features reordered to match model training")
        return reordered
    except KeyError as e:
        raise ValueError(f"Cannot reorder features — missing columns: {e}")
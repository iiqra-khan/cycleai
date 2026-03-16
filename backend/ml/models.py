# =============================================================================
# models.py — Model definitions and class imbalance handling
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


def build_models(y_train: pd.Series = None) -> dict:
    """
    Returns dict of models to train and evaluate.
    All models handle class imbalance natively — NO SMOTE needed.
    """
    if y_train is not None:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 10
        print(f"[LightGBM] scale_pos_weight = {scale_pos_weight:.2f} "
              f"(neg={neg}, pos={pos})")
    else:
        scale_pos_weight = None

    return {
        "Logistic Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.1,
                solver="lbfgs",
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            scale_pos_weight=scale_pos_weight if scale_pos_weight else 1,
            is_unbalance=True if scale_pos_weight is None else False,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
    }


def handle_class_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "class_weight"
):
    """
    Handles class imbalance without resampling temporal/lag features.

    Strategies:
      - class_weight  : no resampling (recommended)
      - oversample    : repeat minority rows
      - smote_safe    : SMOTE on non-temporal features only
    """
    pos_rate = y_train.mean()
    neg      = (y_train == 0).sum()
    pos      = (y_train == 1).sum()

    print(f"\n[Class Balance]")
    print(f"  Total samples:  {len(y_train)}")
    print(f"  Positive (1):   {pos}  ({pos_rate:.3f})")
    print(f"  Negative (0):   {neg}  ({1 - pos_rate:.3f})")
    print(f"  Strategy:       {strategy}")

    if strategy == "class_weight":
        print("  → No resampling. Models use class_weight/scale_pos_weight internally.")
        print("  → Temporal features (lags, rolling) are preserved exactly.")
        return X_train, y_train

    elif strategy == "oversample":
        if pos_rate < 0.15:
            minority_X     = X_train[y_train == 1]
            minority_y     = y_train[y_train == 1]
            repeat_factor  = int((neg / pos) * 0.5)
            X_minority_rep = pd.concat([minority_X] * repeat_factor, ignore_index=True)
            y_minority_rep = pd.concat([minority_y] * repeat_factor, ignore_index=True)
            X_bal = pd.concat([X_train, X_minority_rep], ignore_index=True)
            y_bal = pd.concat([y_train, y_minority_rep], ignore_index=True)
            print(f"  → After oversampling: {len(X_bal)} samples | "
                  f"Positive rate: {y_bal.mean():.3f}")
            return X_bal, y_bal
        else:
            print("  → Balance acceptable, no oversampling needed.")
            return X_train, y_train

    elif strategy == "smote_safe":
        if pos_rate >= 0.15:
            print("  → Balance acceptable, skipping SMOTE.")
            return X_train, y_train

        temporal_keywords = ["lag1", "lag2", "rolling", "momentum", "surge"]
        temporal_cols     = [c for c in X_train.columns
                             if any(kw in c for kw in temporal_keywords)]
        nontemporal_cols  = [c for c in X_train.columns
                             if c not in temporal_cols]

        k  = min(5, pos - 1)
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_nontemp_bal, y_bal = sm.fit_resample(X_train[nontemporal_cols], y_train)

        n_synthetic    = len(X_nontemp_bal) - len(X_train)
        minority_temp  = X_train[temporal_cols][y_train == 1]
        extra_temporal = minority_temp.sample(n=n_synthetic, replace=True, random_state=42)

        X_temporal_bal = pd.concat(
            [X_train[temporal_cols], extra_temporal], ignore_index=True
        )
        X_bal = pd.concat([
            pd.DataFrame(X_nontemp_bal, columns=nontemporal_cols),
            X_temporal_bal.reset_index(drop=True)
        ], axis=1)[X_train.columns]

        print(f"  → After safe SMOTE: {len(X_bal)} samples | "
              f"Positive rate: {pd.Series(y_bal).mean():.3f}")
        return X_bal, pd.Series(y_bal)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose: 'class_weight', 'oversample', 'smote_safe'"
        )
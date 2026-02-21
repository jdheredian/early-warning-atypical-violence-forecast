"""
PASO 6 — Estandarización sin leakage temporal
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_val, X_test, feature_cols):
    """
    Estandariza con media y desviación aprendidas SOLO en train.

    Retorna
    -------
    X_train_sc, X_val_sc, X_test_sc : pd.DataFrame
    scaler : StandardScaler ajustado
    """
    scaler = StandardScaler()

    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols, index=X_train.index
    )
    X_val_sc = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_cols, index=X_val.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols, index=X_test.index
    )

    mean_err = X_train_sc.mean().abs().max()
    std_err  = (X_train_sc.std() - 1).abs().max()
    print(f"✓ Estandarización sin leakage")
    print(f"  Verificación train → Media ~0: {mean_err:.2e} | Std ~1: {std_err:.2e}")

    return X_train_sc, X_val_sc, X_test_sc, scaler

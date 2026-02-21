"""
PASO 5 — Imputación de valores faltantes (sin leakage)
"""
import numpy as np
from sklearn.impute import SimpleImputer


def impute_missing(
    X_train, X_val, X_test,
    numeric_strategy='median',
    categorical_strategy='most_frequent',
):
    """
    Imputa NaN usando estadísticas aprendidas SOLO en train.

    Parámetros
    ----------
    X_train, X_val, X_test : pd.DataFrame
    numeric_strategy : str
        Estrategia para columnas numéricas: 'mean', 'median', 'most_frequent'.
    categorical_strategy : str
        Estrategia para columnas categóricas: 'most_frequent', 'constant'.

    Retorna
    -------
    X_train, X_val, X_test : pd.DataFrame (imputados)
    imputers : dict  {'numeric': ..., 'categorical': ...}
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    total_missing = X_train.isnull().sum().sum()

    print(f"Columnas numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")
    print(f"Valores faltantes en train: {total_missing:,}")

    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    imputers = {}

    if num_cols and X_train[num_cols].isnull().sum().sum() > 0:
        imp = SimpleImputer(strategy=numeric_strategy)
        X_train[num_cols] = imp.fit_transform(X_train[num_cols])
        X_val[num_cols]   = imp.transform(X_val[num_cols])
        X_test[num_cols]  = imp.transform(X_test[num_cols])
        imputers['numeric'] = imp
        print(f"✓ Imputación numérica ({numeric_strategy})")

    if cat_cols and X_train[cat_cols].isnull().sum().sum() > 0:
        imp = SimpleImputer(strategy=categorical_strategy)
        X_train[cat_cols] = imp.fit_transform(X_train[cat_cols])
        X_val[cat_cols]   = imp.transform(X_val[cat_cols])
        X_test[cat_cols]  = imp.transform(X_test[cat_cols])
        imputers['categorical'] = imp
        print(f"✓ Imputación categórica ({categorical_strategy})")

    if total_missing == 0:
        print("✓ Sin valores faltantes, no se requiere imputación")

    remaining = (X_train.isnull().sum().sum()
                 + X_val.isnull().sum().sum()
                 + X_test.isnull().sum().sum())
    print(f"NaNs restantes: {remaining}")

    return X_train, X_val, X_test, imputers

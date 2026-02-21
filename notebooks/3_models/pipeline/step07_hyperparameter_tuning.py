"""
PASO 7 — Búsqueda de hiperparámetros con GridSearchCV + TimeSeriesSplit
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


DEFAULT_PARAM_GRID = {
    'C':        [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
}


def tune_hyperparameters(
    X_train_val,
    y_train_val,
    param_grid=None,
    n_cv_splits=5,
    scoring='f1',
    random_state=42,
):
    """
    Ajusta un Elastic Net con búsqueda de grilla y validación cruzada temporal.

    Parámetros
    ----------
    X_train_val : pd.DataFrame
        Train + validación concatenados (ya estandarizados).
    y_train_val : pd.Series
    param_grid : dict, optional
        Grid de C y l1_ratio. Usa DEFAULT_PARAM_GRID si no se especifica.
    n_cv_splits : int
        Folds de TimeSeriesSplit.
    scoring : str
        Métrica de optimización: 'f1', 'average_precision', 'roc_auc', etc.
    random_state : int

    Retorna
    -------
    best_model : LogisticRegression
    cv_results : pd.DataFrame  (todos los resultados del grid)
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    n_combos = len(param_grid['C']) * len(param_grid['l1_ratio'])
    print(f"GridSearchCV: {n_combos} combinaciones × {n_cv_splits} folds | métrica: {scoring}")

    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=10000,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        tol=1e-4, 
    )

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=n_cv_splits),
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    gs.fit(X_train_val, y_train_val)

    print(f"\n✓ Mejores hiperparámetros:")
    for k, v in gs.best_params_.items():
        print(f"  {k}: {v}")
    print(f"  Mejor {scoring} (CV): {gs.best_score_:.4f}")

    cv_results = pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')
    print(f"\nTop 5 configuraciones:")
    print(cv_results[['param_C', 'param_l1_ratio',
                       'mean_test_score', 'std_test_score']].head(5).to_string(index=False))

    return gs.best_estimator_, cv_results

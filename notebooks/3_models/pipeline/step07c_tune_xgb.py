"""
PASO 7c — Búsqueda de hiperparámetros para XGBoost
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


DEFAULT_PARAM_GRID_XGB = {
    'n_estimators':     [100, 300, 500],
    'max_depth':        [3, 5, 7],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}


def tune_xgb(
    X_train_val,
    y_train_val,
    param_grid=None,
    n_cv_splits=5,
    scoring='f1',
    random_state=42,
    scale_pos_weight=None,
):
    """
    Ajusta un XGBClassifier con búsqueda de grilla y validación cruzada temporal.

    Parámetros
    ----------
    X_train_val : pd.DataFrame
        Train + validación concatenados.
    y_train_val : pd.Series
    param_grid : dict, optional
        Usa DEFAULT_PARAM_GRID_XGB si no se especifica.
    n_cv_splits : int
    scoring : str
    random_state : int
    scale_pos_weight : float or None
        Si None, se calcula automáticamente como n_neg / n_pos.

    Retorna
    -------
    best_model : XGBClassifier
    cv_results : pd.DataFrame
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID_XGB

    from xgboost import XGBClassifier

    if scale_pos_weight is None:
        n_neg = int((y_train_val == 0).sum())
        n_pos = int((y_train_val == 1).sum())
        scale_pos_weight = n_neg / n_pos

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"GridSearchCV XGB: {n_combos} combinaciones × {n_cv_splits} folds | métrica: {scoring}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        enable_categorical=True,
        tree_method='hist',
        n_jobs=-1,
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
    param_cols = [c for c in cv_results.columns if c.startswith('param_')]
    print(f"\nTop 5 configuraciones:")
    print(cv_results[param_cols + ['mean_test_score', 'std_test_score']].head(5).to_string(index=False))

    return gs.best_estimator_, cv_results

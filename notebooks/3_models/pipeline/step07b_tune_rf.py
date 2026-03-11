"""
PASO 7b — Búsqueda de hiperparámetros para Random Forest
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


DEFAULT_PARAM_GRID_RF = {
    'n_estimators':    [100, 300, 500],
    'max_depth':       [5, 10, 20, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf':  [1, 2],
}


def tune_rf(
    X_train_val,
    y_train_val,
    param_grid=None,
    n_cv_splits=5,
    scoring='f1',
    random_state=42,
    class_weight='balanced',
):
    """
    Ajusta un Random Forest con búsqueda de grilla y validación cruzada temporal.

    Parámetros
    ----------
    X_train_val : pd.DataFrame
        Train + validación concatenados.
    y_train_val : pd.Series
    param_grid : dict, optional
        Usa DEFAULT_PARAM_GRID_RF si no se especifica.
    n_cv_splits : int
    scoring : str
    random_state : int
    class_weight : str, dict or None
        Peso de clases. 'balanced' ajusta inversamente a frecuencias.

    Retorna
    -------
    best_model : RandomForestClassifier
    cv_results : pd.DataFrame
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID_RF

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"GridSearchCV RF: {n_combos} combinaciones × {n_cv_splits} folds | métrica: {scoring}")
    print(f"  class_weight: {class_weight}")

    model = RandomForestClassifier(
        class_weight=class_weight,
        random_state=random_state,
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

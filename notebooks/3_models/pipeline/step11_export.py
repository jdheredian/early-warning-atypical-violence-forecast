"""
PASO 11 — Exportar resultados
"""
import os
import pandas as pd
from pathlib import Path
from datetime import datetime


def export_results(
    results_dir,
    experiment_name,
    coeficientes,
    extra_info=None,
):
    """
    Guarda coeficientes / feature importance como CSV con timestamp.

    Parámetros
    ----------
    results_dir : str | Path
    experiment_name : str
    coeficientes : pd.DataFrame
        Resultado de get_coefficients() o get_feature_importance().
    extra_info : dict, optional

    Retorna
    -------
    paths : dict  {'coeficientes': ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    coef_path = os.path.join(results_dir, f"{experiment_name}_coeficientes_{ts}.csv")
    coeficientes.to_csv(coef_path, index=False)

    print(f"✓ Coeficientes exportados: {os.path.basename(coef_path)}")

    return {'coeficientes': coef_path}


def export_predictions(
    results_dir,
    experiment_name,
    model_type,
    target_col,
    mun_codes,
    quarters,
    y_true,
    y_pred,
    y_proba,
    threshold,
):
    """
    Exporta predicciones a nivel observación para el notebook de evaluación.

    Parámetros
    ----------
    results_dir : str | Path
    experiment_name : str
        Nombre codificado del experimento (ej: 'en_iacv', 'rf_t01').
    model_type : str
        Tipo de modelo: 'en', 'rf', 'xgb'.
    target_col : str
        Nombre de la variable dependiente.
    mun_codes : pd.Series
        Códigos municipales del conjunto de test.
    quarters : pd.Series
        Trimestres del conjunto de test.
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Predicciones binarias.
    y_proba : array-like
        Probabilidades predichas.
    threshold : float

    Retorna
    -------
    path : str
    """
    os.makedirs(results_dir, exist_ok=True)

    df_pred = pd.DataFrame({
        'mun_code': mun_codes.values,
        'quarter': quarters.values,
        'y_true': y_true.values if hasattr(y_true, 'values') else y_true,
        'y_pred': y_pred,
        'y_proba': y_proba,
    })
    df_pred['model'] = model_type
    df_pred['target'] = target_col
    df_pred['experiment'] = experiment_name
    df_pred['threshold'] = threshold

    path = Path(results_dir) / f'{experiment_name}_predictions.parquet'
    df_pred.to_parquet(path, index=False)

    print(f"✓ Predicciones exportadas: {path.name}  ({len(df_pred):,} obs)")

    return str(path)

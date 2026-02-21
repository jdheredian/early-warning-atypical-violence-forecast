"""
PASO 11 — Exportar resultados (métricas y coeficientes)
"""
import os
import pandas as pd
from datetime import datetime


def export_results(
    results_dir,
    experiment_name,
    metrics,
    coeficientes,
    extra_info=None,
):
    """
    Guarda métricas y coeficientes como CSV con timestamp.

    Parámetros
    ----------
    results_dir : str | Path
        Carpeta de salida (se crea si no existe).
    experiment_name : str
        Prefijo para los nombres de archivo.
    metrics : dict
        Resultados de evaluate_model().
    coeficientes : pd.DataFrame
        Resultados de get_coefficients().
    extra_info : dict, optional
        Campos adicionales a añadir al CSV de métricas.

    Retorna
    -------
    paths : dict  {'metrics': ..., 'coeficientes': ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    row = {'Experimento': experiment_name, 'Timestamp': ts, **metrics}
    if extra_info:
        row.update(extra_info)

    metrics_path = os.path.join(results_dir, f"{experiment_name}_metrics_{ts}.csv")
    coef_path    = os.path.join(results_dir, f"{experiment_name}_coeficientes_{ts}.csv")

    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    coeficientes.to_csv(coef_path, index=False)

    print(f"✓ Resultados exportados en: {results_dir}")
    print(f"  Métricas     : {os.path.basename(metrics_path)}")
    print(f"  Coeficientes : {os.path.basename(coef_path)}")

    return {'metrics': metrics_path, 'coeficientes': coef_path}

"""
Pipeline de Elastic Net para clasificación de violencia atípica
===============================================================

Pasos disponibles
-----------------
01  load_data               — Carga y prepara el dataset
02  analyze_target          — Distribución de la variable objetivo
03  create_temporal_splits  — Splits train / val / test sin overlap
04  prepare_features        — Selección de X e y
05  impute_missing          — Imputación sin leakage
06  scale_features          — Estandarización sin leakage
07  tune_hyperparameters    — GridSearchCV con TimeSeriesSplit
08  optimize_threshold      — Threshold óptimo en validación
09  evaluate_model          — Métricas finales en test
10  get_coefficients        — Interpretabilidad del modelo
11  export_results          — Exportar CSVs de métricas y coeficientes

Uso rápido
----------
>>> from pipeline import (
...     load_data, analyze_target, create_temporal_splits,
...     prepare_features, impute_missing, scale_features,
...     tune_hyperparameters, optimize_threshold,
...     evaluate_model, get_coefficients, export_results,
... )
"""

from .step01_data_loading         import load_data
from .step02_target_analysis      import analyze_target
from .step03_temporal_splits      import create_temporal_splits
from .step04_feature_preparation  import prepare_features
from .step05_imputation           import impute_missing
from .step06_scaling              import scale_features
from .step07_hyperparameter_tuning import tune_hyperparameters
from .step08_threshold_optimization import optimize_threshold
from .step09_evaluation           import evaluate_model
from .step10_interpretability     import get_coefficients
from .step11_export               import export_results

__all__ = [
    "load_data",
    "analyze_target",
    "create_temporal_splits",
    "prepare_features",
    "impute_missing",
    "scale_features",
    "tune_hyperparameters",
    "optimize_threshold",
    "evaluate_model",
    "get_coefficients",
    "export_results",
]

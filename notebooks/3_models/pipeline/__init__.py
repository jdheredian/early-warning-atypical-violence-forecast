"""
Pipeline de clasificación de violencia atípica
===============================================

Pasos disponibles
-----------------
01   load_data               — Carga y prepara el dataset
02   analyze_target          — Distribución de la variable objetivo
03   create_temporal_splits  — Splits train / val / test sin overlap
04   prepare_features        — Selección de X e y
05   impute_missing          — Imputación sin leakage
05b  undersample_train       — Random undersampling (solo train)
06   scale_features          — Estandarización sin leakage
07   tune_hyperparameters    — GridSearchCV Elastic Net
07b  tune_rf                 — GridSearchCV Random Forest
07c  tune_xgb                — GridSearchCV XGBoost
08   optimize_threshold      — Threshold óptimo en validación
09   evaluate_model          — Métricas finales en test
10   get_coefficients        — Interpretabilidad Elastic Net
10b  get_feature_importance  — Importancia de features (árboles)
11   export_results          — Exportar CSVs de métricas y coeficientes

Uso rápido
----------
>>> from pipeline import (
...     load_data, analyze_target, create_temporal_splits,
...     prepare_features, impute_missing, scale_features,
...     tune_hyperparameters, tune_rf, tune_xgb,
...     optimize_threshold, evaluate_model,
...     get_coefficients, get_feature_importance, export_results,
... )
"""

from .step01_data_loading         import load_data
from .step02_target_analysis      import analyze_target
from .step03_temporal_splits      import create_temporal_splits
from .step04_feature_preparation  import prepare_features
from .step05_imputation           import impute_missing
from .step05b_undersampling       import undersample_train
from .step06_scaling              import scale_features
from .step07_hyperparameter_tuning import tune_hyperparameters
from .step07b_tune_rf             import tune_rf
from .step07c_tune_xgb            import tune_xgb
from .step08_threshold_optimization import optimize_threshold
from .step09_evaluation           import evaluate_model
from .step10_interpretability     import get_coefficients
from .step10b_feature_importance  import get_feature_importance
from .step11_export               import export_results, export_predictions

__all__ = [
    "load_data",
    "analyze_target",
    "create_temporal_splits",
    "prepare_features",
    "impute_missing",
    "undersample_train",
    "scale_features",
    "tune_hyperparameters",
    "tune_rf",
    "tune_xgb",
    "optimize_threshold",
    "evaluate_model",
    "get_coefficients",
    "get_feature_importance",
    "export_results",
    "export_predictions",
]

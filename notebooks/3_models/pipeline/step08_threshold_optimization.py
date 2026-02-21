"""
PASO 8 — Optimización de threshold en validación
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score


def optimize_threshold(
    model,
    X_val_scaled,
    y_val,
    threshold_range=(0.1, 0.9),
    threshold_step=0.05,
    optimize_metric='f1',
    plot=True,
):
    """
    Encuentra el threshold que maximiza una métrica sobre el conjunto de validación.

    Parámetros
    ----------
    model : modelo ajustado con predict_proba.
    X_val_scaled : pd.DataFrame  (ya estandarizado).
    y_val : pd.Series
    threshold_range : tuple  (min, max)
    threshold_step : float
    optimize_metric : str  'f1' | 'precision' | 'recall'
    plot : bool

    Retorna
    -------
    best_threshold : float
    threshold_df : pd.DataFrame  con columnas threshold, f1, precision, recall.
    """
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_step)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            'threshold': t,
            'f1':        f1_score(y_val, y_pred, zero_division=0),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall':    recall_score(y_val, y_pred, zero_division=0),
        })

    threshold_df = pd.DataFrame(rows)
    best_idx = threshold_df[optimize_metric].idxmax()
    best_threshold = threshold_df.loc[best_idx, 'threshold']
    best_row = threshold_df.loc[best_idx]

    print(f"Threshold óptimo ({optimize_metric}): {best_threshold:.2f}")
    print(f"  F1: {best_row['f1']:.4f} | "
          f"Precision: {best_row['precision']:.4f} | "
          f"Recall: {best_row['recall']:.4f}")

    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(threshold_df['threshold'], threshold_df['f1'],
                'o-', label='F1', linewidth=2)
        ax.plot(threshold_df['threshold'], threshold_df['precision'],
                's--', label='Precision', alpha=0.7)
        ax.plot(threshold_df['threshold'], threshold_df['recall'],
                '^--', label='Recall', alpha=0.7)
        ax.axvline(best_threshold, color='red', linestyle=':',
                   linewidth=2, label=f'Óptimo: {best_threshold:.2f}')
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default: 0.50')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Optimización de Threshold', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return best_threshold, threshold_df

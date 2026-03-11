"""
PASO 10b — Importancia de features para modelos basados en árboles
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_feature_importance(model, feature_cols, top_n=20, plot=True):
    """
    Extrae e inspecciona la importancia de features de un modelo de árboles.

    Parámetros
    ----------
    model : objeto sklearn/xgboost con atributo .feature_importances_
    feature_cols : list[str]
    top_n : int
        Cantidad de features a mostrar.
    plot : bool

    Retorna
    -------
    df_importance : pd.DataFrame  (columnas: Feature, Importance), ordenado desc.
    """
    importances = model.feature_importances_

    df_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    non_zero = df_imp.query("Importance > 0")

    print(f"\nFeatures con importancia > 0: {len(non_zero)} / {len(feature_cols)}")
    print(f"\nTop {min(top_n, len(non_zero))} features:")
    print(non_zero[['Feature', 'Importance']].head(top_n).to_string(index=False))

    if plot and len(non_zero) > 0:
        top = non_zero.head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        y_pos = range(len(top) - 1, -1, -1)
        ax.barh(list(y_pos), top['Importance'].values, color='steelblue')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top['Feature'].values)
        ax.set_xlabel('Importancia')
        ax.set_title(f'Top {len(top)} features por importancia')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return df_imp

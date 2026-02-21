"""
PASO 10 — Interpretabilidad: coeficientes del Elastic Net
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_coefficients(model, feature_cols, top_n=20, plot=True):
    """
    Extrae los coeficientes del modelo y genera un ranking de importancia.

    Parámetros
    ----------
    model : LogisticRegression ajustado.
    feature_cols : list[str]
    top_n : int  — cuántas features mostrar en el gráfico.
    plot : bool

    Retorna
    -------
    coeficientes : pd.DataFrame
        Columnas: Feature, Coeficiente, Abs_Coef, Odds_Ratio.
        Ordenado de mayor a menor magnitud.
    """
    coeficientes = pd.DataFrame({
        'Feature':     feature_cols,
        'Coeficiente': model.coef_[0],
        'Abs_Coef':    np.abs(model.coef_[0]),
    }).sort_values('Abs_Coef', ascending=False).reset_index(drop=True)

    coeficientes['Odds_Ratio'] = np.exp(coeficientes['Coeficiente'])

    non_zero = coeficientes[coeficientes['Coeficiente'] != 0]
    zeroed   = coeficientes[coeficientes['Coeficiente'] == 0]

    print(f"{'='*60}")
    print("INTERPRETABILIDAD — COEFICIENTES")
    print(f"{'='*60}")
    print(f"  Features activas    : {len(non_zero)} de {len(feature_cols)}")
    print(f"  Eliminadas (coef=0) : {len(zeroed)}")
    if not zeroed.empty:
        print(f"  → {zeroed['Feature'].tolist()}")

    n_show = min(top_n, len(non_zero))
    print(f"\nTop {n_show} por magnitud:")
    print(non_zero[['Feature', 'Coeficiente', 'Odds_Ratio']].head(n_show).to_string(index=False))

    if plot and not non_zero.empty:
        top = non_zero.head(top_n).sort_values('Coeficiente')
        colors = ['coral' if c > 0 else 'steelblue' for c in top['Coeficiente']]

        fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.38)))
        ax.barh(top['Feature'], top['Coeficiente'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Coeficiente', fontsize=12)
        ax.set_title(f'Top {len(top)} Features — Elastic Net',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()

    return coeficientes

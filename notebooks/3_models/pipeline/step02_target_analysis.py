"""
PASO 2 — Análisis de la variable objetivo
"""
import numpy as np
import matplotlib.pyplot as plt


def analyze_target(df, target_col, time_group_col='año', plot=True):
    """
    Analiza la distribución de la variable dependiente binaria.

    Parámetros
    ----------
    df : pd.DataFrame
    target_col : str
        Variable objetivo (binaria 0/1).
    time_group_col : str
        Columna para agrupar la evolución temporal (default: 'año').
    plot : bool
        Si True, genera gráficos de distribución y evolución temporal.

    Retorna
    -------
    stats : dict
        prevalence, n_positive, n_negative, imbalance_ratio.
    """
    prevalence = df[target_col].mean()
    n_pos = int(df[target_col].sum())
    n_neg = len(df) - n_pos
    imbalance = n_neg / n_pos if n_pos > 0 else float('inf')

    print(f"{'='*60}")
    print(f"ANÁLISIS DE VARIABLE OBJETIVO: '{target_col}'")
    print(f"{'='*60}")
    print(f"  Clase 1 (positivo): {n_pos:,}  ({prevalence:.2%})")
    print(f"  Clase 0 (negativo): {n_neg:,}  ({1 - prevalence:.2%})")
    print(f"  Desbalance        : {imbalance:.2f}:1")

    if prevalence < 0.05:
        print("⚠️  Desbalance EXTREMO (<5%). Considerar SMOTE u otras técnicas.")
    elif prevalence < 0.15:
        print("⚠️  Desbalance significativo (<15%). class_weight='balanced' es crítico.")

    if plot and time_group_col in df.columns:
        temporal = df.groupby(time_group_col)[target_col].agg(
            Casos_Positivos='sum', Total_Obs='count', Proporcion='mean'
        )
        temporal['Proporcion_%'] = temporal['Proporcion'] * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        temporal['Proporcion_%'].plot(kind='bar', ax=ax1, color='coral')
        ax1.axhline(prevalence * 100, color='red', linestyle='--', linewidth=2,
                    label=f'Media global: {prevalence*100:.1f}%')
        ax1.set_title(f"Evolución de '{target_col}' por {time_group_col}",
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel(time_group_col)
        ax1.set_ylabel('% Casos Positivos')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        df[target_col].value_counts().plot(
            kind='pie', ax=ax2, autopct='%1.1f%%',
            labels=['Clase 0', 'Clase 1'], colors=['lightblue', 'coral']
        )
        ax2.set_title('Distribución Global de Clases', fontsize=14, fontweight='bold')
        ax2.set_ylabel('')

        plt.tight_layout()
        plt.show()

    return dict(prevalence=prevalence, n_positive=n_pos,
                n_negative=n_neg, imbalance_ratio=imbalance)

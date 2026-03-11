"""
PASO 5b — Random undersampling de la clase mayoritaria (solo en train)
"""
import numpy as np


def undersample_train(X_train, y_train, ratio=1.0, random_state=42):
    """
    Reduce aleatoriamente la clase mayoritaria en el conjunto de entrenamiento.

    Parámetros
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento (ya imputados).
    y_train : pd.Series
        Target binario de entrenamiento.
    ratio : float
        Ratio deseado n_mayoritaria / n_minoritaria.
        1.0 = equilibrio perfecto (1:1).
        2.0 = dos negativos por cada positivo.
        Valores mayores conservan más observaciones de la clase mayoritaria.
    random_state : int

    Retorna
    -------
    X_train_us : pd.DataFrame
    y_train_us : pd.Series
    """
    n_minority = int(y_train.sum())
    n_majority_target = int(n_minority * ratio)

    idx_minority = y_train[y_train == 1].index
    idx_majority = y_train[y_train == 0].index

    # Si el ratio pedido supera los datos disponibles, usar todos
    n_majority_target = min(n_majority_target, len(idx_majority))

    rng = np.random.default_rng(random_state)
    idx_majority_sampled = rng.choice(
        idx_majority, size=n_majority_target, replace=False
    )

    idx_final = np.concatenate([idx_minority, idx_majority_sampled])

    X_train_us = X_train.loc[idx_final].copy()
    y_train_us = y_train.loc[idx_final].copy()

    prev_antes  = y_train.mean()
    prev_despues = y_train_us.mean()

    print(f"Undersampling (ratio mayoría:minoría = {ratio:.1f}:1):")
    print(f"  antes  → {len(y_train):,} obs | prevalencia: {prev_antes:.2%}")
    print(f"  después → {len(y_train_us):,} obs | prevalencia: {prev_despues:.2%}")
    print(f"  clase 0: {int((y_train_us == 0).sum()):,} | clase 1: {int((y_train_us == 1).sum()):,}")

    return X_train_us, y_train_us

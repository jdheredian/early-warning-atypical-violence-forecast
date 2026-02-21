"""
PASO 4 — Separación de features y target

Dos modos de selección de variables:
  - Por exclusión  : pasar cols_to_exclude  (se excluyen esas + las obligatorias).
  - Por selección  : pasar feature_cols     (lista explícita de X a usar).
"""


def prepare_features(
    df,
    target_col,
    municipality_col,
    time_col,
    cols_to_exclude=None,
    feature_cols=None,
):
    """
    Construye X e y separando features y target.

    Parámetros
    ----------
    df : pd.DataFrame
    target_col : str
        Variable dependiente.
    municipality_col : str
        Identificador de municipio (siempre excluido de X).
    time_col : str
        Columna temporal (siempre excluida de X).
    cols_to_exclude : list[str], optional
        Columnas adicionales a excluir de X (además de target, ids y derivadas).
    feature_cols : list[str], optional
        Lista explícita de features. Si se provee, ignora cols_to_exclude.

    Retorna
    -------
    X : pd.DataFrame
    y : pd.Series
    feature_cols : list[str]
    """
    # Siempre excluidas
    always_exclude = {target_col, municipality_col, time_col,
                      'año', 'trimestre', 'periodo_num'}

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Features no encontradas en el dataset: {missing}")
    else:
        all_exclude = always_exclude | set(cols_to_exclude or [])
        not_found = [c for c in all_exclude if c not in df.columns]
        if not_found:
            print(f"⚠️  Columnas de exclusión no encontradas (ignoradas): {not_found}")
        feature_cols = [c for c in df.columns if c not in all_exclude]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"{'='*60}")
    print(f"FEATURES DEL MODELO")
    print(f"{'='*60}")
    print(f"  Features seleccionadas : {len(feature_cols)}")
    print(f"  Target                 : '{target_col}'  (prevalencia: {y.mean():.2%})")
    print(f"\n  Variables:")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:3d}. {col}")

    return X, y, feature_cols

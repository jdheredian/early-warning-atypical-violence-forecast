"""
PASO 1 — Carga y preparación de datos
"""
import pandas as pd


def load_data(data_path, time_col='quarter', municipality_col='mun_code'):
    """
    Carga datos desde un parquet, extrae año/trimestre y ordena cronológicamente.

    Parámetros
    ----------
    data_path : str | Path
        Ruta al archivo .parquet.
    time_col : str
        Columna temporal en formato 'YYYYQN' (ej: '2006Q1').
    municipality_col : str
        Columna de identificador de municipio.

    Retorna
    -------
    df : pd.DataFrame
        Dataset ordenado con columnas 'año', 'trimestre' y 'periodo_num' añadidas.
    """
    print(f"Cargando datos desde: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"✓ Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # Columnas temporales derivadas
    df['año'] = df[time_col].astype(str).str[:4].astype(int)
    df['trimestre'] = df[time_col].astype(str).str[-1].astype(int)
    df['periodo_num'] = df['año'] * 10 + df['trimestre']

    df = df.sort_values(['periodo_num', municipality_col]).reset_index(drop=True)

    print(f"Rango temporal : {df[time_col].min()} → {df[time_col].max()}")
    print(f"Periodos únicos: {df[time_col].nunique()} | Municipios: {df[municipality_col].nunique()}")
    return df

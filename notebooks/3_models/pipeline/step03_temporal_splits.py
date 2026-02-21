"""
PASO 3 — Splits temporales sin overlap ni leakage
"""


def create_temporal_splits(
    df,
    time_col,
    train_prop=0.70,
    val_prop=0.15,
    test_prop=0.15,
    use_year_splits=False,
    train_end_year=None,
    train_end_quarter=None,
    val_end_year=None,
    val_end_quarter=None,
):
    """
    Crea máscaras de train / validación / test respetando el orden temporal.

    Modos
    -----
    - Proporciones (default): divide los periodos únicos según train_prop / val_prop / test_prop.
    - Años específicos      : usa train_end_year/quarter y val_end_year/quarter como cortes.

    Retorna
    -------
    splits : dict
        Llaves: train_mask, val_mask, test_mask, train_periods, val_periods, test_periods.
    """
    periodos = sorted(df[time_col].unique())
    n = len(periodos)

    if use_year_splits:
        tc = f"{train_end_year}Q{train_end_quarter}"
        vc = f"{val_end_year}Q{val_end_quarter}"
        train_periods = [p for p in periodos if p <= tc]
        val_periods   = [p for p in periodos if tc < p <= vc]
        test_periods  = [p for p in periodos if p > vc]
        train_mask = df[time_col] <= tc
        val_mask   = (df[time_col] > tc) & (df[time_col] <= vc)
        test_mask  = df[time_col] > vc
        print(f"✓ Splits por años: Train ≤ {tc} | Val ≤ {vc} | Test > {vc}")
    else:
        ti = int(n * train_prop)
        vi = int(n * (train_prop + val_prop))
        train_periods = periodos[:ti]
        val_periods   = periodos[ti:vi]
        test_periods  = periodos[vi:]
        train_mask = df[time_col].isin(train_periods)
        val_mask   = df[time_col].isin(val_periods)
        test_mask  = df[time_col].isin(test_periods)
        print(f"✓ Splits por proporciones ({train_prop:.0%}/{val_prop:.0%}/{test_prop:.0%}):")

    for name, mask, periods in [
        ('Train', train_mask, train_periods),
        ('Val  ', val_mask,   val_periods),
        ('Test ', test_mask,  test_periods),
    ]:
        print(f"  {name}: {len(periods):2d} periodos  "
              f"({periods[0]} → {periods[-1]})  "
              f"{mask.sum():,} obs ({mask.sum()/len(df):.1%})")

    # Validaciones
    assert (train_mask & val_mask).sum()  == 0, "❌ Overlap Train/Val"
    assert (train_mask & test_mask).sum() == 0, "❌ Overlap Train/Test"
    assert (val_mask & test_mask).sum()   == 0, "❌ Overlap Val/Test"
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == len(df), "❌ Obs. faltantes"
    print("✓ Sin overlap")

    return dict(
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        train_periods=train_periods, val_periods=val_periods, test_periods=test_periods,
    )

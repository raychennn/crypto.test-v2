import pandas as pd
import numpy as np

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(length).mean()

def get_slope(series: pd.Series, window: int = 48) -> pd.Series:
    """
    用簡單線性回歸斜率近似「趨勢斜率」。
    window=48 在 1H 下約 2 天。
    """
    x = np.arange(window)
    def slope(y):
        if len(y) < window:
            return np.nan
        # polyfit: y = a*x + b
        a = np.polyfit(x, y, 1)[0]
        return a
    return series.rolling(window).apply(slope, raw=True)

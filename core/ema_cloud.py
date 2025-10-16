import pandas as pd
def ema_cloud(candles: pd.DataFrame, short=20, mid=50):
    df = candles.copy()
    df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['ema_mid'] = df['close'].ewm(span=mid, adjust=False).mean()
    # simple cloud direction:
    df['cloud_state'] = df.apply(lambda r: 'flat' if abs(r['ema_short']-r['ema_mid'])/r['ema_mid'] < 0.002 else ('up' if r['ema_short']>r['ema_mid'] else 'down'), axis=1)
    return df[['ema_short','ema_mid','cloud_state']]
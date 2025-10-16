import ccxt, yfinance as yf
import pandas as pd
from core.utils import RAW_DIR
from pathlib import Path

def fetch_binance_ohlcv(symbol="BTC/USDT", timeframe="1h", since_days=120):
    """
    Fetch recent OHLCV from Binance (public). Saves parquet under data/raw/.
    """
    ex = ccxt.binance()
    limit = 1000
    ms_now = ex.milliseconds()
    since = ms_now - int(since_days*24*3600*1000)
    all_rows = []
    t = since
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=t, limit=limit)
        if not ohlcv:
            break
        all_rows += ohlcv
        t = ohlcv[-1][0] + 1
        if t >= ms_now:
            break
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('dt').drop(columns=['ts'])
    p = RAW_DIR / f"{symbol.replace('/','-')}_{timeframe}.parquet"
    df.to_parquet(p)
    return p

def fetch_yfinance(symbol="^GSPC", period="1y", interval="1h"):
    df = yf.download(symbol, period=period, interval=interval)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    p = RAW_DIR / f"{symbol.replace('^','').replace('.','')}_{interval}.parquet"
    df.to_parquet(p)
    return p
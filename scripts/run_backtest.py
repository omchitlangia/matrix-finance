import argparse, pandas as pd
from core.data_fetcher import fetch_binance_ohlcv, fetch_yfinance
from core.volume_profile import compute_volume_profile, find_peaks, find_overlapping_levels
from core.backtester import simple_backtest
from core.utils import RAW_DIR
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    args=parser.parse_args()
    # user should have data in data/raw; fetch if not
    p = RAW_DIR / f"{args.symbol.replace('/','-')}_{args.timeframe}.parquet"
    if not p.exists():
        print("Fetching historical candles (may take a while)...")
        p = fetch_binance_ohlcv(symbol=args.symbol, timeframe=args.timeframe, since_days=120)
    candles = pd.read_parquet(p)
    # VRVP compute on last 200 candles
    vrvp = compute_volume_profile(candles.tail(200), n_bins=140)
    # SVP per session (group by date)
    candles['date'] = candles.index.date
    sessions=[]
    for date, g in candles.groupby('date'):
        svp = compute_volume_profile(g, n_bins=140)
        peaks = find_peaks(svp['vol'], svp['centers'], top_n=3)
        sessions.append({"date":str(date), "peaks":peaks})
    levels = find_overlapping_levels(vrvp, sessions, tolerance_pct=0.002)
    print("Found levels:", levels)
    # For demo we need a sentiment_time_series: create zeros if none
    import numpy as np
    sentiment_ts = pd.DataFrame({"agg_score": np.zeros(len(candles))}, index=candles.index)
    trades = simple_backtest(candles, levels, sentiment_ts)
    print("Trades:", trades)
if __name__=="__main__":
    main()
import pandas as pd
import numpy as np
from core.signal_generator import generate_signals
from core.ema_cloud import ema_cloud

def simple_backtest(candles: pd.DataFrame, levels, sentiment_time_series, entry_tol=0.0015, pos_size_usd=10.0, fee_pct=0.0006):
    """
    Simple event-driven backtest:
      - On each candle (1h), check levels; if price near level and cloud/sentiment conditions -> enter.
      - Exit when SL/TP hit or opposite signal occurs or after max_hold
    sentiment_time_series: DataFrame with timestamp index & column 'agg_score' (hourly aligned)
    """
    candles = candles.copy()
    candles['dt'] = candles.index
    res_trades=[]
    open_pos=None
    for idx,row in candles.iterrows():
        # get sentiment aligned
        srow = sentiment_time_series.loc[sentiment_time_series.index <= idx].tail(1)
        sentiment = float(srow['agg_score'].values[0]) if not srow.empty else 0.0
        cloud = ema_cloud(candles.loc[:idx])[['cloud_state']].iloc[-1]
        cand_slice = candles.loc[:idx]
        signals = generate_signals(cand_slice, levels, sentiment, ema_cloud(cand_slice))
        # if open pos -> check exit
        price_open = row['open']; price_high = row['high']; price_low = row['low']; price_close = row['close']
        if open_pos:
            side = open_pos['side']
            sl = open_pos['sl']; tp = open_pos['tp']
            exit=False; exit_price=None; reason=None
            if side=='long':
                if price_low <= sl:
                    exit=True; exit_price=sl; reason='SL'
                elif price_high >= tp:
                    exit=True; exit_price=tp; reason='TP'
            else:
                if price_high >= sl:
                    exit=True; exit_price=sl; reason='SL'
                elif price_low <= tp:
                    exit=True; exit_price=tp; reason='TP'
            if exit:
                pnl = (exit_price - open_pos['entry']) * open_pos['units'] if side=='long' else (open_pos['entry'] - exit_price) * open_pos['units']
                pnl -= open_pos['fees'] + abs(exit_price*open_pos['units'])*fee_pct
                res_trades.append({**open_pos, "exit_time":idx, "exit_price":exit_price, "pnl":pnl, "exit_reason":reason})
                open_pos=None
        # if no open pos and there is a signal -> open first signal (simple)
        if (not open_pos) and signals:
            sig = signals[0]
            entry_price = sig['entry']  # we assume market at close->open behavior may be adjusted
            units = pos_size_usd / entry_price
            fees = pos_size_usd * fee_pct
            open_pos = {"entry":entry_price, "side":sig['side'], "units":units, "sl":sig['sl'], "tp":sig['tp'], "entry_time":idx, "fees":fees}
    trades_df = pd.DataFrame(res_trades)
    return trades_df
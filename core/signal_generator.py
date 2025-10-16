import pandas as pd
from core.volume_profile import compute_volume_profile, find_peaks, vrvp_zones, find_overlapping_levels
from core.ema_cloud import ema_cloud
from core.sentiment_engine import aggregate_and_save, score_text
import numpy as np

def price_near_level(price, level_price, tol_pct=0.0015):  # ~0.15%
    return abs(price - level_price) <= level_price * tol_pct

def generate_signals(candles: pd.DataFrame, levels: list, sentiment_score: float, cloud_df: pd.DataFrame,
                     entry_tol=0.0015, sentiment_threshold=0.05):
    """
    candles: candles to evaluate (must include final bar as the decision bar)
    levels: list of dict {'price':..., 'vr_zone':(...)}
    cloud_df: output of ema_cloud for same candles
    returns: list of signals with entry price, side, sl, tp, reason
    """
    latest = candles.iloc[-1]
    price = float(latest['close'])
    cloud_state = cloud_df.iloc[-1]['cloud_state']
    signals=[]
    for lvl in levels:
        p = lvl['price']
        if price_near_level(price, p, entry_tol):
            # determine direction: if cloud up => prefer long on support; if cloud down => prefer short on resistance
            side = None
            if cloud_state == 'up':
                # prefer longs near level below price (support)
                if price >= p:  # price at or above level can be bounce or break; we allow buy if level acted as support
                    side = 'long'
            elif cloud_state == 'down':
                if price <= p:
                    side = 'short'
            else:
                continue
            # sentiment filter: require sentiment to be aligned (sentiment_score positive for long etc)
            if side == 'long' and sentiment_score < -sentiment_threshold:
                continue
            if side == 'short' and sentiment_score > sentiment_threshold:
                continue
            # compute sl/tp: naive: sl = nearest opposite level, tp = next level in direction
            # for demo, we set sl as p +/- (p*0.01) buffer (user can improve)
            if side == 'long':
                sl = p - max(0.5, p*0.005)
                # TP find next higher level if exists else p + ATR-like buffer
                higher = [l for l in levels if l['price'] > p]
                tp = higher[0]['price'] if higher else p + (p*0.01)
            else:
                sl = p + max(0.5, p*0.005)
                lower = [l for l in levels if l['price'] < p]
                tp = lower[-1]['price'] if lower else p - (p*0.01)
            signals.append({"side":side, "level":p, "entry":price, "sl":float(sl), "tp":float(tp), "cloud":cloud_state, "sentiment":sentiment_score})
    return signals
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from core.utils import VP_CACHE, save_json
from pathlib import Path

def price_bins_from_range(low, high, n_bins=120):
    return np.linspace(low, high, n_bins+1)

def compute_volume_profile(candles: pd.DataFrame, n_bins=120, distribute_volume=False):
    """
    Compute a volume histogram across price bins.
    If distribute_volume=False: assign each candle's volume to the bin of its close (fast).
    If distribute_volume=True: distribute candle volume across bins intersecting candle high-low (more accurate).
    Returns dict with edges, centers, vol array.
    """
    lo = float(candles['low'].min())
    hi = float(candles['high'].max())
    edges = price_bins_from_range(lo, hi, n_bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vol = np.zeros(n_bins, dtype=float)
    if not distribute_volume:
        idx = np.searchsorted(edges, candles['close'].values, side='right') - 1
        idx = np.clip(idx, 0, n_bins-1)
        for i,v in zip(idx, candles['volume'].values):
            vol[i] += v
    else:
        # distribute each candle's volume proportionally across bins overlapped by high-low
        bin_width = edges[1]-edges[0]
        for r in candles.itertuples():
            low_idx = int(np.clip(np.searchsorted(edges, r.low) - 1, 0, n_bins-1))
            high_idx = int(np.clip(np.searchsorted(edges, r.high) - 1, 0, n_bins-1))
            if high_idx < low_idx:
                high_idx = low_idx
            counts = high_idx - low_idx + 1
            if counts <= 0:
                counts = 1
            share = r.volume / counts
            vol[low_idx:high_idx+1] += share
    vol_sm = gaussian_filter1d(vol, sigma=max(1, n_bins//80))
    return {"edges":edges.tolist(), "centers":centers.tolist(), "vol":vol_sm.tolist()}

def find_peaks(vol_arr, centers, top_n=5):
    arr = np.array(vol_arr)
    peaks = []
    for i in range(1,len(arr)-1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            peaks.append((i, arr[i]))
    if not peaks:
        return []
    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"price": float(centers[i]), "vol":float(v)} for i,v in peaks_sorted]

def vrvp_zones(vrvp, pct_cut=0.6):
    vol = np.array(vrvp['vol'])
    centers = np.array(vrvp['centers'])
    cutoff = vol.max() * pct_cut
    high_bins = np.where(vol >= cutoff)[0]
    if len(high_bins)==0:
        return []
    zones=[]
    start=high_bins[0]; prev=start
    for b in high_bins[1:]:
        if b==prev+1:
            prev=b
        else:
            left = float(centers[start] - (centers[1]-centers[0])/2)
            right = float(centers[prev] + (centers[1]-centers[0])/2)
            zones.append((left,right))
            start=b; prev=b
    left = float(centers[start] - (centers[1]-centers[0])/2)
    right = float(centers[prev] + (centers[1]-centers[0])/2)
    zones.append((left,right))
    return zones

def find_overlapping_levels(vrvp, sessions_svp_peaks, tolerance_pct=0.002):
    """
    sessions_svp_peaks: list of {date:..., peaks:[{'price','vol'},...]}
    returns list of levels where SVP peaks overlap VRVP zones
    """
    vr_z = vrvp_zones(vrvp)
    levels=[]
    for rec in sessions_svp_peaks:
        for p in rec['peaks']:
            price = p['price']
            tol = price * tolerance_pct
            for (lo,hi) in vr_z:
                if (price >= lo - tol) and (price <= hi + tol):
                    levels.append({"price": price, "svp_vol": p['vol'], "session": rec['date'], "vr_zone":(lo,hi)})
                    break
    # merge close levels
    levels_sorted = sorted(levels, key=lambda x:x['price'])
    merged=[]
    eps = 1e-6
    for lvl in levels_sorted:
        if not merged:
            merged.append(lvl)
        else:
            if abs(merged[-1]['price'] - lvl['price']) <= max(merged[-1]['price']*0.0005, 0.5): # small tolerance
                # merge by averaging price weighted by svp_vol
                a = merged[-1]; b = lvl
                wp = (a['price']*a['svp_vol'] + b['price']*b['svp_vol']) / (a['svp_vol']+b['svp_vol']+eps)
                merged[-1]['price'] = float(wp)
                merged[-1]['svp_vol'] += b['svp_vol']
            else:
                merged.append(lvl)
    return merged
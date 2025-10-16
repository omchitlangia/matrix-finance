import os, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VP_CACHE = DATA_DIR / "vp_cache"
SENT_DIR = DATA_DIR / "sentiment"

for d in (DATA_DIR, RAW_DIR, VP_CACHE, SENT_DIR):
    os.makedirs(d, exist_ok=True)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def load_parquet_if_exists(path):
    if Path(path).exists():
        return pd.read_parquet(path)
    return None
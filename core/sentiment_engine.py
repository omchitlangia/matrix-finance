import requests, time, os, json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from core.utils import SENT_DIR, save_json
import praw
import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np

sia = SentimentIntensityAnalyzer()

# --- News via NewsAPI (free tier) ---
def fetch_news_newsapi(api_key, q="market OR bitcoin OR crypto OR economy", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {"q":q, "language":"en", "pageSize":page_size, "sortBy":"publishedAt", "apiKey":api_key}
    r = requests.get(url, params=params)
    items = []
    if r.status_code==200:
        for art in r.json().get('articles',[]):
            items.append({"timestamp":art.get("publishedAt"), "text": art.get("title") + " " + (art.get("description") or ""), "source":"newsapi"})
    return items

# --- Reddit via PRAW (requires credentials) ---
def fetch_reddit(client_id, client_secret, user_agent, subs=["CryptoCurrency","Bitcoin","economics"], limit=100):
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    items=[]
    for s in subs:
        for post in reddit.subreddit(s).new(limit=limit):
            items.append({"timestamp": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(), "text": post.title + " " + (post.selftext or ""), "source":"reddit"})
    return items

def score_text(text):
    return sia.polarity_scores(text)['compound']

def aggregate_and_save(newsapi_key=None, reddit_creds=None):
    items=[]
    if newsapi_key:
        items += fetch_news_newsapi(newsapi_key)
    if reddit_creds:
        items += fetch_reddit(**reddit_creds)
    # score
    df = pd.DataFrame(items)
    if df.empty:
        return None
    df['score'] = df['text'].apply(lambda t: score_text(str(t)))
    # assign source weights
    w_map = {'newsapi':0.5, 'reddit':0.3, 'twitter':0.2}
    df['source_w'] = df['source'].map(w_map).fillna(0.2)
    # time decay
    now = pd.Timestamp.utcnow()
    df['ts'] = pd.to_datetime(df['timestamp'])
    df['minutes_ago'] = (now - df['ts']).dt.total_seconds()/60.0
    decay = 180.0
    df['time_w'] = np.exp(-df['minutes_ago']/decay)
    df['w'] = df['source_w'] * df['time_w']
    agg = (df['score'] * df['w']).sum() / (df['w'].sum()+1e-9)
    out = {"timestamp": str(now), "agg_score": float(agg), "n_items": int(len(df))}
    save_json(SENT_DIR / f"sent_{now.strftime('%Y%m%d%H%M%S')}.json", out)
    return out
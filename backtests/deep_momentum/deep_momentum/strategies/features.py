# features.py
import pandas as pd
import numpy as np

def get_features_and_targets(prices: pd.DataFrame):
    features = pd.DataFrame(index=prices.stack().index)
    
    for w in [30, 90, 180, 365]:
        features[f'mom_{w}d'] = prices.pct_change(w).stack()

    features['vol_30d'] = prices.pct_change().rolling(30).std().stack()
    
    future_returns = prices.pct_change(30).shift(-30).stack()
    
    data = features.copy()
    data['target'] = future_returns
    
    return data.dropna()
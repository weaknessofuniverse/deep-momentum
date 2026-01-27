# ml_momentum.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from quant_pml.strategies.factors.sorting_strategy import SortingStrategy
from .features import get_features_and_targets

class MLMomentum(SortingStrategy):
    def __init__(
        self,
        model_type="ridge",
        quantile=0.1,
        mode="long_short",
        **kwargs
    ):
        super().__init__(quantile=quantile, mode=mode, **kwargs)
        
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        
        if model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        elif model_type == "mlp":
            self.model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42)

    def _fit(self, training_data) -> None:
        self._seen_data = training_data
        
        prices = training_data.prices
        
        df = get_features_and_targets(prices)
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"[{self.model_type.upper()}] Fitting on {len(X)} samples...")
        
        with np.errstate(under='ignore', over='ignore'):
            self.model.fit(X_scaled, y)
            
        print(f"[{self.model_type.upper()}] Model fitted successfully.")

    def get_scores(self, data) -> pd.Series:
        if self.model is None:
            return pd.Series(0, index=data.prices.columns)

        lookback = 400
        recent_prices = data.prices.iloc[-lookback:]

        df = get_features_and_targets(recent_prices)

        last_date = df.index.get_level_values(0).max()
        current_features = df[df.index.get_level_values(0) == last_date]
        
        if current_features.empty:
            return pd.Series(0, index=data.prices.columns)
            
        X_current = current_features.drop(columns=['target'], errors='ignore')

        X_scaled = self.scaler.transform(X_current)
        scores = self.model.predict(X_scaled)
        
        assets = current_features.index.get_level_values(1)
        
        return pd.Series(scores, index=assets)
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_pml.strategies.optimization_data import TrainingData

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from quant_pml.strategies.factors.sorting_strategy import SortingStrategy


class _MLPClassifier(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 8, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPClassifierMomentum(SortingStrategy):
    def __init__(  # noqa: PLR0913
        self,
        mode: str,
        sign: int = 1,
        *,
        quantile: float | None = None,
        n_holdings: int | None = None,
        weighting_scheme: str = "equally_weighted",
        train_months: int = 60,
        hidden: int = 64,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 4096,
        epochs_first: int = 10,
        epochs_update: int = 2,
        exclude_td: int = 21,
    ) -> None:
        super().__init__(
            quantile=quantile,
            mode=mode,
            n_holdings=n_holdings,
            weighting_scheme=weighting_scheme,
        )
        self.sign = sign
        self.train_months = train_months
        self.batch_size = batch_size
        self.epochs_first = epochs_first
        self.epochs_update = epochs_update
        self.exclude_td = exclude_td

        self.device = torch.device("cpu")
        self.model = _MLPClassifier(in_dim=3, hidden=hidden, dropout=dropout).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self._trained_once = False
        self._train_dates: list[pd.Timestamp] = []
        self._X_cache: list[np.ndarray] = []
        self._y_cache: list[np.ndarray] = []
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    @staticmethod
    def _month_ends(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        s = pd.Series(dates, index=dates)
        return s.groupby(dates.to_period("M")).max().tolist()

    def _momentum_features_at(self, returns: pd.DataFrame, pos: int) -> pd.DataFrame:
        exc = self.exclude_td

        def window_sum(win: int) -> pd.Series:
            sl = returns.iloc[pos - win - exc : pos - exc]
            return sl.sum(axis=0, min_count=int(0.8 * win))

        r12 = window_sum(252)
        r6 = window_sum(126)
        r3 = window_sum(63)

        X = pd.concat([r12, r6, r3], axis=1)
        X.columns = ["r12", "r6", "r3"]
        return X

    def _next_month_target(self, returns: pd.DataFrame, pos_t: int, pos_next: int) -> pd.Series:
        sl = returns.iloc[pos_t + 1 : pos_next + 1]
        minc = int(0.8 * len(sl)) if len(sl) > 0 else 1
        return sl.sum(axis=0, min_count=minc)

    def _rebuild_scaler(self, X: np.ndarray) -> None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        self._x_mean = mu
        self._x_std = sd

    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        if self._x_mean is None or self._x_std is None:
            return X
        return (X - self._x_mean) / self._x_std

    def _train_one_round(self, epochs: int) -> None:
        X = np.vstack(self._X_cache)
        y_raw = np.concatenate(self._y_cache)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y_raw)
        X = X[mask].astype(np.float32)
        y_raw = y_raw[mask].astype(np.float32)

        if len(X) < 5000:
            return

        y = (y_raw > 0).astype(np.float32)

        self._rebuild_scaler(X)
        X = self._transform_X(X).astype(np.float32)

        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X).to(self.device),
            torch.from_numpy(y).to(self.device),
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                logits = self.model(xb)
                loss = criterion(logits, yb)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def _fit(self, training_data: TrainingData) -> None:
        super()._fit(training_data)

        returns = training_data.simple_total_returns
        if returns is None or len(returns) < (252 + self.exclude_td + 5):
            return

        month_ends = self._month_ends(returns.index)
        if len(month_ends) < 3:
            return

        t_prev = month_ends[-2]
        t_cur = month_ends[-1]

        if t_prev in self._train_dates:
            return

        pos_prev = returns.index.get_loc(t_prev)
        pos_cur = returns.index.get_loc(t_cur)

        X_df = self._momentum_features_at(returns, pos_prev).reindex(index=returns.columns)
        y_ser = self._next_month_target(returns, pos_prev, pos_cur).reindex(index=returns.columns)

        self._train_dates.append(t_prev)
        self._X_cache.append(X_df.to_numpy())
        self._y_cache.append(y_ser.to_numpy())

        if len(self._train_dates) > self.train_months:
            self._train_dates.pop(0)
            self._X_cache.pop(0)
            self._y_cache.pop(0)

        if not self._trained_once:
            self._train_one_round(self.epochs_first)
            self._trained_once = True
        else:
            self._train_one_round(self.epochs_update)

    def get_scores(self, data: TrainingData) -> pd.Series:
        returns = data.simple_total_returns
        if returns is None or len(returns) < (252 + self.exclude_td + 5) or not self._trained_once:
            r12 = returns.iloc[-252 - 21 : -21].sum(axis=0)
            r6 = returns.iloc[-126 - 21 : -21].sum(axis=0)
            r3 = returns.iloc[-63 - 21 : -21].sum(axis=0)
            scores = 0.5 * r12 + 0.3 * r6 + 0.2 * r3
            return self.sign * scores

        pos_cur = len(returns.index) - 1
        X_cur = self._momentum_features_at(returns, pos_cur).reindex(index=returns.columns).to_numpy()
        X_cur = self._transform_X(X_cur).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_cur).to(self.device)
            logits = self.model(x_t).cpu().numpy()

        proba = 1.0 / (1.0 + np.exp(-logits))
        scores = pd.Series(proba, index=returns.columns)
        return self.sign * scores

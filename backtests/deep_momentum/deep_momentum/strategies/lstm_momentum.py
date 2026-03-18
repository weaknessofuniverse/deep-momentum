from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_pml.strategies.optimization_data import TrainingData

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from quant_pml.strategies.factors.sorting_strategy import SortingStrategy


class _LSTM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_dim)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class LSTMMomentum(SortingStrategy):
    MACD_PAIRS = [
        (8, 24), (16, 48), (32, 96),
        (8, 48), (8, 96), (16, 96),
        (16, 24), (32, 48),
    ]

    def __init__(  # noqa: PLR0913
        self,
        mode: str,
        sign: int = 1,
        *,
        quantile: float | None = None,
        n_holdings: int | None = None,
        weighting_scheme: str = "equally_weighted",
        train_months: int = 60,
        seq_len: int = 12,
        hidden: int = 64,
        num_layers: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 8192,
        epochs_first: int = 15,
        exclude_td: int = 21,
        use_macd: bool = True,
    ) -> None:
        super().__init__(
            quantile=quantile,
            mode=mode,
            n_holdings=n_holdings,
            weighting_scheme=weighting_scheme,
        )
        self.sign = sign
        self.train_months = train_months
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs_first = epochs_first
        self.exclude_td = exclude_td
        self.use_macd = use_macd

        self.device = torch.device("cpu")
        self.hidden = hidden
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.in_dim = 24 if use_macd else 8

        self.model: _LSTM | None = None
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
        r1 = window_sum(21)

        vol = returns.iloc[pos - 63 : pos].std(axis=0) * np.sqrt(252)
        vol = vol.replace(0, np.nan)

        r12_norm = r12 / (vol / np.sqrt(252 / 252))
        r6_norm = r6 / (vol / np.sqrt(252 / 126))
        r3_norm = r3 / (vol / np.sqrt(252 / 63))

        X = pd.concat([r12, r6, r3, r1, r12_norm, r6_norm, r3_norm, vol], axis=1)
        X.columns = ["r12", "r6", "r3", "r1", "r12_norm", "r6_norm", "r3_norm", "vol"]
        return X

    def _macd_features_at(self, returns: pd.DataFrame, pos: int) -> pd.DataFrame:
        lookback = 400
        start = max(0, pos - lookback)
        prices = (1 + returns.iloc[start : pos + 1]).cumprod()

        unique_spans = {s for pair in self.MACD_PAIRS for s in pair}
        ema_cache = {span: prices.ewm(span=span, adjust=False).mean() for span in unique_spans}

        cur_price = prices.iloc[-1].replace(0, np.nan)
        features = {}

        for fast, slow in self.MACD_PAIRS:
            macd_series = ema_cache[fast] - ema_cache[slow]
            macd_val = macd_series.iloc[-1] / cur_price
            signal_period = 9
            signal_val = macd_series.ewm(span=signal_period, adjust=False).mean().iloc[-1] / cur_price
            features[f"macd_{fast}_{slow}"] = macd_val
            features[f"signal_{fast}_{slow}"] = signal_val

        return pd.DataFrame(features)

    def _compute_features(self, returns: pd.DataFrame, pos: int) -> pd.DataFrame:
        mom = self._momentum_features_at(returns, pos)
        if not self.use_macd:
            return mom
        macd = self._macd_features_at(returns, pos)
        return pd.concat([mom, macd.reindex(mom.index)], axis=1)

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

    def _train_from_scratch(self) -> None:
        n = len(self._X_cache)
        if n < self.seq_len + 1:
            return

        seq_len = self.seq_len
        in_dim = self.in_dim

        X_seqs = []
        y_list = []
        for j in range(seq_len, n):
            # X_seq: [n_assets, seq_len, in_dim]
            stack = np.stack([self._X_cache[j - seq_len + k] for k in range(seq_len)], axis=1)
            y_j = self._y_cache[j - 1]
            X_seqs.append(stack)
            y_list.append(y_j)

        # X_all: [n_months * n_assets, seq_len, in_dim], y_all: [n_months * n_assets]
        X_all = np.concatenate(X_seqs, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        # Flatten for scaling: [N, seq_len, in_dim] -> treat as N*seq_len samples for mean/std
        N, S, D = X_all.shape
        X_flat = X_all.reshape(-1, D)
        mask_flat = np.isfinite(X_flat).all(axis=1)
        valid_X = X_flat[mask_flat]
        if len(valid_X) < 1000:
            return

        self._rebuild_scaler(valid_X)
        X_all = self._transform_X(X_all.reshape(-1, D)).reshape(N, S, D).astype(np.float32)
        y_all = y_all.astype(np.float32)

        mask = np.isfinite(X_all).all(axis=(1, 2)) & np.isfinite(y_all)
        X_all = X_all[mask]
        y_all = y_all[mask]

        if len(X_all) < 5000:
            return

        self.model = _LSTM(
            in_dim=in_dim,
            hidden=self.hidden,
            num_layers=self.num_layers,
        ).to(self.device)
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        crit = nn.SmoothL1Loss()

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_all).to(self.device),
            torch.from_numpy(y_all).to(self.device),
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model.train()
        for _ in range(self.epochs_first):
            for xb, yb in dl:
                pred = self.model(xb)
                loss = crit(pred, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()

        self._trained_once = True

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

        X_df = self._compute_features(returns, pos_prev).reindex(index=returns.columns)
        y_ser = self._next_month_target(returns, pos_prev, pos_cur).reindex(index=returns.columns)

        X = X_df.to_numpy()
        y = y_ser.to_numpy()

        self._train_dates.append(t_prev)
        self._X_cache.append(X)
        self._y_cache.append(y)

        if len(self._train_dates) > self.train_months:
            self._train_dates.pop(0)
            self._X_cache.pop(0)
            self._y_cache.pop(0)

        self._train_from_scratch()

    def get_scores(self, data: TrainingData) -> pd.Series:
        returns = data.simple_total_returns
        if returns is None or len(returns) < (252 + self.exclude_td + 5) or not self._trained_once:
            r12 = returns.iloc[-252 - 21 : -21].sum(axis=0)
            r6 = returns.iloc[-126 - 21 : -21].sum(axis=0)
            r3 = returns.iloc[-63 - 21 : -21].sum(axis=0)
            scores = 0.5 * r12 + 0.3 * r6 + 0.2 * r3
            return self.sign * scores

        month_ends = self._month_ends(returns.index)
        if len(month_ends) < self.seq_len:
            r12 = returns.iloc[-252 - 21 : -21].sum(axis=0)
            r6 = returns.iloc[-126 - 21 : -21].sum(axis=0)
            r3 = returns.iloc[-63 - 21 : -21].sum(axis=0)
            scores = 0.5 * r12 + 0.3 * r6 + 0.2 * r3
            return self.sign * scores

        last_12 = month_ends[-self.seq_len :]
        X_list = []
        for t in last_12:
            pos = returns.index.get_loc(t)
            X_t = self._compute_features(returns, pos).reindex(index=returns.columns).to_numpy()
            X_list.append(X_t)

        X_seq = np.stack(X_list, axis=1)
        X_seq = self._transform_X(X_seq.reshape(-1, self.in_dim)).reshape(
            X_seq.shape[0], self.seq_len, self.in_dim
        ).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_seq).to(self.device)
            pred = self.model(x_t).cpu().numpy()

        scores = pd.Series(pred, index=returns.columns)
        return self.sign * scores

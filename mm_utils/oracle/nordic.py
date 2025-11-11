from math import sqrt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from util.util import log_print


class NordicLOBOracle:
    """Oracle that derives a fundamental price path from the Nordic LOB benchmark dataset.

    The dataset (FI-2010) ships as transposed feature matrices where each row corresponds to a
    feature stream across all samples.  We only require the top-of-book prices to construct a
    fundamental series, but we retain best bid/ask paths for potential downstream usage.
    """

    def __init__(
        self,
        data_file,
        symbol,
        mkt_open,
        mkt_close=None,
        freq="100ms",
        price_scale=10000.0,
        price_offset=2.0,
        enforce_monotonic=True,
    ):
        self.data_file = Path(data_file)
        self.symbol = symbol
        self.freq = pd.to_timedelta(freq)
        self.price_scale = float(price_scale)
        self.price_offset = float(price_offset)
        self.enforce_monotonic = enforce_monotonic

        self.mkt_open = pd.Timestamp(mkt_open)
        self.mkt_close_hint = pd.Timestamp(mkt_close) if mkt_close is not None else None

        self._mid_series = None
        self._bid_series = None
        self._ask_series = None

        self._load_dataset()

    # ---------------------------------------------------------------------
    def _load_dataset(self):
        if not self.data_file.exists():
            raise FileNotFoundError(f"Nordic LOB dataset not found: {self.data_file}")

        log_print("NordicLOBOracle: loading {}", self.data_file)
        matrix = np.loadtxt(self.data_file)

        # Dataset ships as (features, samples).  If the opposite, transpose to match.
        if matrix.shape[0] < matrix.shape[1]:
            features = matrix
        else:
            features = matrix.T

        # Bid/ask price levels (top 10) alternate with corresponding volumes.
        bid_price_levels = features[0:20:2]
        ask_price_levels = features[20:40:2]

        best_bid_norm = bid_price_levels[0]
        best_ask_norm = ask_price_levels[0]

        bid_price = (best_bid_norm + self.price_offset) * self.price_scale
        ask_price = (best_ask_norm + self.price_offset) * self.price_scale

        if self.enforce_monotonic:
            mask = ask_price <= bid_price
            ask_price[mask] = bid_price[mask] + 1.0

        mid_price = (bid_price + ask_price) / 2.0

        index = pd.date_range(start=self.mkt_open, periods=len(mid_price), freq=self.freq)

        self._bid_series = pd.Series(np.round(bid_price).astype(int), index=index)
        self._ask_series = pd.Series(np.round(ask_price).astype(int), index=index)
        self._mid_series = pd.Series(np.round(mid_price).astype(int), index=index)

        if self.mkt_close_hint is not None and index[-1] < self.mkt_close_hint:
            log_print(
                "NordicLOBOracle: generated series ends before hinted close ({} < {}).",
                index[-1],
                self.mkt_close_hint,
            )

        log_print(
            "NordicLOBOracle: constructed price path with {} samples spanning {} to {}",
            len(self._mid_series),
            index[0],
            index[-1],
        )

    # ---------------------------------------------------------------------
    def _get_series(self, symbol):
        if symbol != self.symbol:
            raise KeyError(f"Oracle only tracks symbol {self.symbol}")
        return self._mid_series

    def getDailyOpenPrice(self, symbol, mkt_open):
        self.mkt_open = pd.Timestamp(mkt_open)
        series = self._get_series(symbol)
        price = int(series.iloc[0])
        log_print("NordicLOBOracle: market open {} price {}", mkt_open, price)
        return price

    def _search_price(self, series, query_time):
        ts = pd.Timestamp(query_time)
        index = series.index
        if ts <= index[0]:
            return int(series.iloc[0])
        if ts >= index[-1]:
            return int(series.iloc[-1])

        pos = index.searchsorted(ts)
        if index[pos] == ts:
            return int(series.iloc[pos])

        low = series.iloc[pos - 1]
        high = series.iloc[pos]
        t_low = index[pos - 1]
        t_high = index[pos]
        frac = (ts - t_low) / (t_high - t_low)
        return int(round(low + (high - low) * frac))

    def getPriceAtTime(self, symbol, query_time):
        series = self._get_series(symbol)
        return self._search_price(series, query_time)

    def observePrice(self, symbol, currentTime, sigma_n=0.0001, random_state=None):
        true_price = float(self.getPriceAtTime(symbol, currentTime))
        if sigma_n == 0 or random_state is None:
            observed = true_price
        else:
            observed = random_state.normal(loc=true_price, scale=sqrt(sigma_n))
        return int(round(observed))

    # Convenience accessors ------------------------------------------------
    def getBestBidAsk(self, symbol, query_time) -> Optional[tuple]:
        if symbol != self.symbol:
            return None
        ts = pd.Timestamp(query_time)
        bid = self._search_price(self._bid_series, ts)
        ask = self._search_price(self._ask_series, ts)
        return bid, ask

    def to_dict(self):
        return {
            "data_file": str(self.data_file),
            "freq": str(self.freq),
            "price_scale": self.price_scale,
            "price_offset": self.price_offset,
            "enforce_monotonic": self.enforce_monotonic,
            "samples": len(self._mid_series),
            "start": str(self._mid_series.index[0]),
            "end": str(self._mid_series.index[-1]),
        }

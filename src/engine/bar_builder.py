"""
bar_builder.py — Aggregate live ticks into M1 OHLCV bars for real-time inference.

LiveBarBuilder accumulates Tick objects arriving from the live stream and,
when the UTC minute advances, produces a completed OHLCV bar row and a
ticks DataFrame ready for micro-structure feature engineering.

This closes the loop identified in the live inference design: without this
component, _process_tick() ran feature engineering on static Parquet data and
the live price stream was only a once-per-minute heartbeat.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ml_engine.model import Tick


class LiveBarBuilder:
    """
    Accumulate live ticks into a completed M1 OHLCV bar.

    One instance lives inside LiveEngine for each (symbol, expiry_key) pair.
    Call add_tick() on every incoming tick. On the minute boundary, call
    close_and_reset() to obtain the completed bar row and the ticks DataFrame.

    The bar row is a plain dict — no Bar dataclass construction needed for the
    inference path. The ticks DataFrame has a DatetimeIndex and bid/ask columns
    matching the interface expected by FeatureEngineer._compute_micro_structure().

    Thread safety: not thread-safe. LiveEngine calls this only from the asyncio
    event loop, sequentially, so no locking is required.
    """

    __slots__ = ("symbol", "_ticks", "_open", "_high", "_low")

    def __init__(self, symbol: str) -> None:
        self.symbol: str = symbol
        self._ticks: list[Tick] = []
        self._open: float | None = None
        self._high: float = -math.inf
        self._low: float = math.inf

    # ── Public interface ───────────────────────────────────────────────────────

    def add_tick(self, tick: Tick) -> None:
        """Update OHLCV state from a single incoming tick."""
        mid = tick.mid_price
        if self._open is None:
            self._open = mid
        self._high = max(self._high, mid)
        self._low = min(self._low, mid)
        self._ticks.append(tick)

    def close_and_reset(
        self,
        ref_index: pd.DatetimeIndex,
    ) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """
        Finalise the current bar, build the ticks DataFrame, and reset.

        Args:
            ref_index: The DatetimeIndex of the historical bars_df.
                Used to match timezone (tz-aware vs naive) so the live bar
                row can be concatenated without alignment warnings.

        Returns:
            (live_bar_df, ticks_df) where:
                live_bar_df: Single-row DataFrame with columns [open, high, low,
                    close, volume] and a DatetimeIndex aligned to ref_index's tz.
                    None if no ticks arrived this bar.
                ticks_df: DataFrame with DatetimeIndex and [bid, ask] columns
                    for _compute_micro_structure(). Empty DataFrame if no ticks.
        """
        ticks = self._ticks
        open_ = self._open
        high = self._high
        low = self._low

        # Reset immediately so subsequent ticks go into the next bar.
        self._ticks = []
        self._open = None
        self._high = -math.inf
        self._low = math.inf

        empty_ticks = pd.DataFrame({"bid": pd.Series(dtype=float), "ask": pd.Series(dtype=float)})

        if not ticks or open_ is None:
            return None, empty_ticks

        # Build ticks DataFrame — DatetimeIndex, bid + ask columns.
        ticks_df = pd.DataFrame(
            {
                "bid": [t.bid for t in ticks],
                "ask": [t.ask for t in ticks],
            },
            index=pd.DatetimeIndex([t.timestamp for t in ticks]),
        )

        # Bar timestamp = start of the minute this bar represents.
        # Use replace() rather than .floor() to avoid pandas-stubs arity issue.
        _t = ticks[0].timestamp
        bar_ts = pd.Timestamp(_t.replace(second=0, microsecond=0))

        # Match timezone of the historical index to avoid concat warnings.
        if ref_index.tz is not None:
            bar_ts = (
                bar_ts.tz_convert(ref_index.tz)
                if bar_ts.tzinfo is not None
                else bar_ts.tz_localize(ref_index.tz)
            )
        else:
            # Strip tz to match a naive historical index.
            bar_ts = bar_ts.replace(tzinfo=None) if bar_ts.tzinfo is not None else bar_ts

        close_price = ticks[-1].mid_price

        live_bar_df = pd.DataFrame(
            {
                "open": [open_],
                "high": [high],
                "low": [low],
                "close": [close_price],
                "volume": [float(len(ticks))],
            },
            index=pd.DatetimeIndex([bar_ts]),
        )

        return live_bar_df, ticks_df

    def __len__(self) -> int:
        return len(self._ticks)

    def __repr__(self) -> str:
        return (
            f"LiveBarBuilder(symbol={self.symbol!r}, "
            f"ticks={len(self._ticks)}, "
            f"open={self._open})"
        )

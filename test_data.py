"""Minimal Phase 1 tests: data structure, chronology and integrity.

These tests read the frozen artifacts written by `download_data.py` and make no
network calls, so they are reproducible offline. Run `python3 download_data.py`
first. The intent is to validate the dataset the paper rests on, not to
exercise the script's internals.

Run:  python3 -m pytest test_data.py -v
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import download_data as dd

PRICES_PATH = Path("data/processed/adjusted_prices.csv")
RETURNS_PATH = Path("data/processed/daily_returns.csv")
MANIFEST_PATH = Path("data/raw/manifest.json")


def _load(path):
    if not path.exists():
        pytest.skip(f"{path} not found; run python3 download_data.py first")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


@pytest.fixture(scope="module")
def prices():
    return _load(PRICES_PATH)


@pytest.fixture(scope="module")
def returns():
    return _load(RETURNS_PATH)


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest not found; run python3 download_data.py first")
    return json.loads(MANIFEST_PATH.read_text())


# --- Structure ---------------------------------------------------------------

def test_panel_has_exactly_the_protocol_instruments(prices):
    assert list(prices.columns) == dd.TICKERS


def test_no_missing_values(prices, returns):
    assert not prices.isna().any().any()
    assert not returns.isna().any().any()


def test_prices_are_strictly_positive(prices):
    assert (prices > 0).all().all()


def test_returns_align_with_prices(prices, returns):
    assert list(returns.columns) == list(prices.columns)
    # One return per price observation, except the leading return base.
    assert len(returns) == len(prices) - 1
    assert returns.index.equals(prices.index[1:])


# --- Chronology --------------------------------------------------------------

def test_index_is_naive_datetime_sorted_and_unique(prices):
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.tz is None
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique


def test_no_weekend_observations(prices):
    assert not (prices.index.dayofweek >= 5).any()


def test_return_sample_covers_the_protocol_window(returns):
    assert str(returns.index.min().date()) == dd.SAMPLE_START
    assert str(returns.index.max().date()) == dd.SAMPLE_END


def test_price_panel_starts_one_trading_day_before_the_sample(prices):
    assert prices.index[0] < pd.Timestamp(dd.SAMPLE_START)
    assert str(prices.index[1].date()) == dd.SAMPLE_START


def test_sample_is_long_enough_for_the_estimation_window(returns):
    # 756-day window plus at least one month of out-of-sample evaluation.
    assert len(returns) > 756 + 21


def test_known_market_closure_is_absent_not_missing(prices):
    # 2007-01-02: NYSE closed for the national day of mourning for President
    # Ford. A real closure, so it must not appear in the panel at all.
    assert pd.Timestamp("2007-01-02") not in prices.index
    assert pd.Timestamp("2007-01-03") in prices.index


# --- Integrity ---------------------------------------------------------------

def test_no_implausible_daily_returns(returns):
    worst = returns.abs().max()
    assert (worst <= dd.MAX_PLAUSIBLE_ABS_DAILY_RETURN).all(), worst.to_dict()


@pytest.mark.parametrize("ticker,date", dd.KNOWN_SPURIOUS_SPLITS)
def test_spurious_split_is_not_applied_to_prices(returns, ticker, date):
    """Yahoo reports a 10:1 SLV split on 2008-07-24 that never happened.

    If it were ever applied, the daily return would be about -0.9 or +9.0.
    """
    stamp = pd.Timestamp(date)
    window = returns[ticker].loc[stamp - pd.Timedelta(days=5): stamp + pd.Timedelta(days=5)]
    assert not window.empty
    assert window.abs().max() < 0.25


def test_no_stale_price_runs(prices):
    """Repeated identical prices would indicate forward filling of closures."""
    for ticker in dd.TICKERS:
        unchanged = (prices[ticker].diff() == 0)
        longest = unchanged.groupby((~unchanged).cumsum()).sum().max()
        assert longest <= 3, f"{ticker} has {longest} consecutive unchanged prices"


def test_raw_archive_matches_its_checksums(manifest):
    for name, info in manifest["files"].items():
        path = dd.RAW_DIR / name
        if not path.exists():
            pytest.skip(f"{name} not archived; checksums only (see protocol U8)")
        assert dd.sha256(path) == info["sha256"], f"{name} does not match its checksum"


def test_manifest_records_provenance(manifest):
    assert manifest["downloaded_utc"]
    assert set(manifest["versions"]) >= {"python", "pandas", "yfinance"}
    assert {i["ticker"] for i in manifest["files"].values()} == set(dd.TICKERS)
    # auto_adjust must stay False so the adjustment can be verified independently.
    assert manifest["download_arguments"]["auto_adjust"] is False

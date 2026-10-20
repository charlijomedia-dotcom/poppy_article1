"""Minimal Phase 1 tests: data structure, chronology and integrity.

These tests read the frozen artifacts written by `download_data.py` and make no
network calls, so they are reproducible offline. Run `python3 download_data.py`
first. A required file that is missing is a failure, not a reason to skip: the
point of the suite is to prove the dataset the paper rests on is present and
sound.

Run:  python3 -m pytest test_data.py -v
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import download_data as dd

PRICES_PATH = Path("data/processed/adjusted_prices.csv")
RETURNS_PATH = Path("data/processed/daily_returns.csv")
RISK_FREE_PATH = Path("data/processed/risk_free_daily.csv")
MANIFEST_PATH = dd.MANIFEST_PATH

_REBUILD_HINT = "run 'python3 download_data.py' to rebuild it from the frozen raw inputs"


def _require(path):
    assert path.exists(), f"required file {path} is missing; {_REBUILD_HINT}"
    return path


def _load(path):
    frame = pd.read_csv(_require(path), index_col="Date", parse_dates=True)
    assert not frame.empty, f"{path} is empty; {_REBUILD_HINT}"
    return frame


@pytest.fixture(scope="module")
def prices():
    return _load(PRICES_PATH)


@pytest.fixture(scope="module")
def returns():
    return _load(RETURNS_PATH)


@pytest.fixture(scope="module")
def risk_free():
    return _load(RISK_FREE_PATH)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(_require(MANIFEST_PATH).read_text())


# --- Required artifacts exist ------------------------------------------------

@pytest.mark.parametrize(
    "path", [PRICES_PATH, RETURNS_PATH, RISK_FREE_PATH, MANIFEST_PATH]
)
def test_required_artifact_exists(path):
    _require(path)


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


def test_panel_matches_the_official_exchange_calendar(returns):
    """Instrument agreement cannot reveal a day absent from all four series."""
    official = dd.exchange_trading_days(dd.SAMPLE_START, dd.SAMPLE_END)
    ours = pd.DatetimeIndex(returns.index).normalize()
    assert list(official.difference(ours)) == []
    assert list(ours.difference(official)) == []


@pytest.mark.parametrize("date,reason", dd.EXCEPTIONAL_CLOSURES)
def test_exceptional_closures_are_absent_from_the_panel(prices, date, reason):
    stamp = pd.Timestamp(date)
    if not (prices.index.min() <= stamp <= pd.Timestamp(dd.SAMPLE_END)):
        pytest.skip(f"{date} lies outside the panel window")
    assert stamp.dayofweek < 5, f"{date} is not a weekday"
    assert stamp not in prices.index, f"{date} was a closure ({reason}) but is in the panel"


# --- Integrity ---------------------------------------------------------------

def test_no_implausible_daily_returns(returns):
    worst = returns.abs().max()
    assert (worst <= dd.MAX_PLAUSIBLE_ABS_DAILY_RETURN).all(), worst.to_dict()


@pytest.mark.parametrize("split", dd.KNOWN_SPLITS, ids=lambda s: f"{s['ticker']}-{s['date']}")
def test_split_is_adjusted_exactly_once(returns, split):
    """Genuine splits are already reflected in Yahoo's price columns.

    SLV's 10-for-1 split of 2008-07-24 is real (SEC Form 10-Q, CIK 0001330568).
    Because `Close` and `Adj Close` are both back-adjusted for it, the series is
    continuous across the date. A missing adjustment would show about -90% and a
    double adjustment about +900%, so continuity is the test that the adjustment
    is present exactly once.
    """
    ticker, date, ratio = split["ticker"], split["date"], split["ratio"]
    stamp = pd.Timestamp(date)
    window = returns[ticker].loc[stamp - pd.Timedelta(days=5): stamp + pd.Timedelta(days=5)]
    assert not window.empty
    assert window.abs().max() < 0.25, (
        f"{ticker} is discontinuous around its {ratio:.0f}-for-1 split of {date}: "
        f"max |return| {window.abs().max():.4f}"
    )


def test_no_stale_price_runs(prices):
    """Repeated identical prices would indicate forward filling of closures."""
    for ticker in dd.TICKERS:
        unchanged = (prices[ticker].diff() == 0)
        longest = unchanged.groupby((~unchanged).cumsum()).sum().max()
        assert longest <= 3, f"{ticker} has {longest} consecutive unchanged prices"


# --- Risk-free series --------------------------------------------------------

def test_risk_free_covers_every_trading_day(returns, risk_free):
    assert risk_free.index.equals(returns.index)


def test_risk_free_has_no_unresolved_gaps(risk_free):
    assert not risk_free[["annualized_percent", "daily_rate"]].isna().any().any()


def test_risk_free_carry_forward_is_limited_and_flagged(risk_free):
    years = len(risk_free) / dd.TRADING_DAYS_PER_YEAR
    carried = int(risk_free["carried_forward"].sum())
    assert carried > 0, "bond-market holidays should be flagged, not silently absent"
    assert carried / years <= dd.MAX_CARRIED_FORWARD_YIELDS_PER_YEAR


def test_risk_free_daily_rate_matches_the_documented_conversion(risk_free):
    expected = (1.0 + risk_free["annualized_percent"] / 100.0) ** (
        1.0 / dd.TRADING_DAYS_PER_YEAR
    ) - 1.0
    assert (risk_free["daily_rate"] - expected).abs().max() < 1e-15
    assert (risk_free["annualized_percent"] >= 0).all()


# --- Frozen inputs -----------------------------------------------------------

def test_frozen_raw_inputs_match_their_checksums(manifest):
    for name, info in manifest["files"].items():
        path = _require(dd.RAW_DIR / name)
        assert dd.sha256(path) == info["sha256"], f"{name} does not match its checksum"


def test_manifest_records_provenance(manifest):
    assert manifest["downloaded_utc"]
    assert set(manifest["versions"]) >= {"python", "pandas", "yfinance"}
    roles = {}
    for info in manifest["files"].values():
        roles.setdefault(info["role"], set()).add(info["key"])
    assert roles["analysis"] == set(dd.TICKERS)
    assert roles["cross_check"] == set(dd.CROSS_CHECK_PEERS.values())
    assert roles["risk_free"] == {dd.RISK_FREE_SERIES}
    # auto_adjust must stay False so the adjustment can be verified independently.
    assert manifest["download_arguments"]["auto_adjust"] is False

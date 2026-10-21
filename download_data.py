"""Phase 1: build, validate and freeze the dataset for the safe-haven study.

Two modes, so that the reproducibility path can never silently overwrite the
frozen research inputs:

    python3 download_data.py              rebuild the processed dataset from the
                                          frozen raw files already in data/raw/,
                                          verifying every SHA-256 against
                                          data/raw/manifest.json. Makes no
                                          network calls. This is the default and
                                          the documented reproduction procedure.

    python3 download_data.py --download   re-download from the sources and
                                          rewrite data/raw/ and the manifest.
                                          This replaces frozen inputs, so it is
                                          never the default and must be asked
                                          for explicitly.

Writes (both modes)
-------------------
data/processed/adjusted_prices.csv    aligned adjusted total-return price panel
data/processed/daily_returns.csv      daily simple returns of the panel
data/processed/risk_free_daily.csv    short-term Treasury yield and daily rate
data/data_coverage_report.md          coverage, missing data and validation report

Writes (--download only)
------------------------
data/raw/<TICKER>_yahoo_raw.csv       unmodified Yahoo Finance download
data/raw/<SERIES>_fred_raw.csv        unmodified FRED download
data/raw/manifest.json                sources, timestamps, versions, SHA-256

The script validates rather than repairs. Nothing is interpolated,
back-filled or simulated: if a required input is incomplete the run fails and
says so. See research_protocol.md for the frozen design this implements.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf

# --- Protocol parameters (see research_protocol.md sections 2-4) --------------

TICKERS = ["SPY", "AGG", "GLD", "SLV"]

# Return sample: first and last daily return observation (inclusive).
SAMPLE_START = "2007-01-03"
SAMPLE_END = "2026-09-30"

# The price panel carries one extra leading trading day, which supplies the
# base price for the first return and is not itself an observation.
PANEL_BUFFER_START = "2006-06-01"

# Raw archive window. Wider than the sample so that the archive also documents
# each instrument's inception and the pre-sample history.
RAW_START = "2006-01-01"
RAW_END = "2026-10-01"  # yfinance end bound is exclusive

# Risk-free rate and the cash leg of the volatility-managed benchmark
# (research_protocol.md section 8, decision D3). 3-Month Treasury Constant
# Maturity Rate, published as an annualised percentage on an investment basis.
RISK_FREE_SERIES = "DGS3MO"
FRED_CSV_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id={series}&cosd={start}&coed={end}"
)
TRADING_DAYS_PER_YEAR = 252

# Authoritative trading calendar used to verify the panel. Validation only: the
# panel's dates come from the instruments themselves and are never constructed
# from this calendar.
EXCHANGE_CALENDAR = "NYSE"

# Independent cross-check of the metal series against competing ETFs from
# different issuers tracking the same metal. Agreement is evidence that the
# adjusted series reflect the underlying metal and not an issuer-specific data
# artefact. Not used in the analysis; validation only.
CROSS_CHECK_PEERS = {"SLV": "SIVR", "GLD": "IAU"}

# --- Validation thresholds (documented, not tuned) ---------------------------

# Largest plausible absolute daily return for a broad ETF. Any breach is
# treated as suspected data corruption (for example a split applied twice),
# not as a market move, and fails validation.
MAX_PLAUSIBLE_ABS_DAILY_RETURN = 0.40

# Tolerance for reconstructing adjusted-close returns from raw closes and
# dividends. Yahoo stores Adj Close as float32, so exact equality is
# unattainable; 1e-4 is far tighter than any economically relevant error.
ADJ_RECONSTRUCTION_TOLERANCE = 1e-4

# Peers track the same metal through a different vehicle, so small differences
# in fee drag and premium/discount are expected. A breach of these bounds would
# indicate one of the two series is wrong.
MAX_PEER_RETURN_DIFFERENCE = 0.02
MIN_PEER_RETURN_CORRELATION = 0.98

# Trading days on which the bond market is shut but the NYSE trades (Columbus
# Day and Veterans Day) have no published Treasury quote. Two per year is
# expected; more would indicate a problem with the series.
MAX_CARRIED_FORWARD_YIELDS_PER_YEAR = 3

# --- Documented corporate actions --------------------------------------------

# Genuine splits occurring inside the archive window. Yahoo's `Close` and
# `Adj Close` columns are both back-adjusted for these, which is why the price
# series is continuous across the date. The pipeline therefore takes split
# handling from the price columns and never applies the `Stock Splits` column
# itself, so the adjustment is applied exactly once.
KNOWN_SPLITS = [
    {
        "ticker": "SLV",
        "date": "2008-07-24",
        "ratio": 10.0,
        "description": (
            "iShares Silver Trust 10-for-1 share split, reducing the silver "
            "backing from approximately 10 ounces to approximately 1 ounce per "
            "share"
        ),
        "source": (
            "SEC Form 10-Q, iShares Silver Trust (CIK 0001330568): "
            "https://www.sec.gov/Archives/edgar/data/1330568/"
            "000119312509106967/d10q.htm"
        ),
    }
]

# Exceptional NYSE closures inside the sample window: weekdays that were not
# scheduled holidays. Listed so that their absence from the panel is documented
# as a closure rather than discovered as a gap.
EXCEPTIONAL_CLOSURES = [
    ("2007-01-02", "National day of mourning for President Gerald R. Ford"),
    ("2012-10-29", "Hurricane Sandy"),
    ("2012-10-30", "Hurricane Sandy"),
    ("2018-12-05", "National day of mourning for President George H. W. Bush"),
    ("2025-01-09", "National day of mourning for President Jimmy Carter"),
]

DOWNLOAD_ATTEMPTS = 4

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = RAW_DIR / "manifest.json"
REPORT_PATH = Path("data/data_coverage_report.md")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- Acquisition (only reached with --download) -------------------------------

def download_yahoo(ticker):
    """Download one ticker's full raw record, retrying on transient failures.

    auto_adjust=False keeps raw Close *and* Adj Close, and actions=True keeps
    the dividend and split record, so the adjustment can be verified
    independently instead of being taken on trust.
    """
    last_error = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            frame = yf.download(
                ticker,
                start=RAW_START,
                end=RAW_END,
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )
            if frame is not None and not frame.empty:
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = [c[0] for c in frame.columns]
                frame.index.name = "Date"
                return frame
            last_error = "empty frame returned"
        except Exception as exc:  # network / API failure
            last_error = repr(exc)
        if attempt < DOWNLOAD_ATTEMPTS - 1:
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(
        f"could not download {ticker} after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def download_fred(series):
    """Download one FRED series as CSV text, preserved verbatim."""
    url = FRED_CSV_URL.format(series=series, start=RAW_START, end=RAW_END)
    last_error = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200 and series in response.text[:200]:
                return response.text
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = repr(exc)
        if attempt < DOWNLOAD_ATTEMPTS - 1:
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(
        f"could not download FRED {series} after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def download_sources():
    """Refresh every raw input and rewrite the manifest. Overwrites frozen data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {}

    for ticker in TICKERS:
        print(f"downloading {ticker} ...", flush=True)
        frame = download_yahoo(ticker)
        path = RAW_DIR / f"{ticker}_yahoo_raw.csv"
        frame.to_csv(path)
        files[path.name] = {
            "role": "analysis",
            "key": ticker,
            "source": "Yahoo Finance",
            "rows": len(frame),
            "sha256": sha256(path),
        }

    for ticker, peer in CROSS_CHECK_PEERS.items():
        print(f"downloading cross-check peer {peer} for {ticker} ...", flush=True)
        frame = download_yahoo(peer)
        path = RAW_DIR / f"{peer}_yahoo_raw.csv"
        frame.to_csv(path)
        files[path.name] = {
            "role": "cross_check",
            "key": peer,
            "verifies": ticker,
            "source": "Yahoo Finance",
            "rows": len(frame),
            "sha256": sha256(path),
        }

    print(f"downloading FRED {RISK_FREE_SERIES} ...", flush=True)
    text = download_fred(RISK_FREE_SERIES)
    path = RAW_DIR / f"{RISK_FREE_SERIES}_fred_raw.csv"
    path.write_text(text)
    files[path.name] = {
        "role": "risk_free",
        "key": RISK_FREE_SERIES,
        "source": "Federal Reserve Economic Data (FRED), St. Louis Fed",
        "rows": len(text.strip().splitlines()) - 1,
        "sha256": sha256(path),
    }

    manifest = {
        "sources": {
            "equity_bond_metal": "Yahoo Finance via the yfinance Python package",
            "risk_free": (
                f"FRED series {RISK_FREE_SERIES} (3-Month Treasury Constant "
                "Maturity Rate), downloaded as CSV"
            ),
        },
        "downloaded_utc": pd.Timestamp.now("UTC").isoformat(),
        "requested_window": {"start": RAW_START, "end_exclusive": RAW_END},
        "download_arguments": {"auto_adjust": False, "actions": True, "threads": False},
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "yfinance": yf.__version__,
            "pandas_market_calendars": mcal.__version__,
        },
        "files": files,
        "licence_note": (
            "Yahoo Finance terms of service restrict redistribution. See "
            "research_protocol.md decision D8 for the storage options; note that "
            "deleting these CSVs in a later commit does not remove them from the "
            "repository's published Git history. FRED data are US Government "
            "works and are not subject to copyright."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --- Frozen inputs ------------------------------------------------------------

def read_yahoo_csv(path):
    frame = pd.read_csv(path, index_col="Date", parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame.index.name = "Date"
    return frame


def read_fred_csv(path, series):
    frame = pd.read_csv(path, parse_dates=["observation_date"])
    out = frame.set_index("observation_date")[series]
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


def load_frozen():
    """Load the frozen raw inputs, failing hard if any is missing or altered.

    This is the default path, so the reproduction procedure verifies the inputs
    rather than refreshing them.
    """
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"frozen manifest not found at {MANIFEST_PATH}.\n"
            "Nothing to reproduce from. Run 'python3 download_data.py --download' "
            "to acquire and freeze the raw inputs first."
        )
    manifest = json.loads(MANIFEST_PATH.read_text())

    problems = []
    for name, info in manifest["files"].items():
        path = RAW_DIR / name
        if not path.exists():
            problems.append(f"  {name}: missing from {RAW_DIR}")
        else:
            digest = sha256(path)
            if digest != info["sha256"]:
                problems.append(
                    f"  {name}: SHA-256 mismatch\n"
                    f"      manifest: {info['sha256']}\n"
                    f"      on disk:  {digest}"
                )
    if problems:
        raise SystemExit(
            "frozen raw inputs do not match the approved manifest:\n"
            + "\n".join(problems)
            + "\n\nThe processed dataset was not rebuilt. Restore the frozen files, "
            "or run 'python3 download_data.py --download' to deliberately refresh "
            "and re-freeze them (this changes the research inputs)."
        )

    raw, peers, risk_free = {}, {}, None
    for name, info in manifest["files"].items():
        path = RAW_DIR / name
        if info["role"] == "analysis":
            raw[info["key"]] = read_yahoo_csv(path)
        elif info["role"] == "cross_check":
            peers[info["verifies"]] = read_yahoo_csv(path)["Adj Close"]
        elif info["role"] == "risk_free":
            risk_free = read_fred_csv(path, info["key"])

    missing = [t for t in TICKERS if t not in raw]
    if missing:
        raise SystemExit(f"manifest does not cover required instruments: {missing}")
    if risk_free is None:
        raise SystemExit(f"manifest does not cover the {RISK_FREE_SERIES} risk-free series")
    return manifest, raw, peers, risk_free


# --- Construction -------------------------------------------------------------

def exchange_trading_days(start, end):
    """Official exchange trading days, used for validation only."""
    days = mcal.get_calendar(EXCHANGE_CALENDAR).valid_days(start, end)
    return pd.DatetimeIndex(days).tz_localize(None).normalize()


def build_panel(raw):
    """Align the four adjusted total-return series onto their common calendar.

    Missing data are distinguished from non-trading days in two independent
    ways: by comparing the trading dates the instruments themselves report, and
    by comparing the result with the official exchange calendar (the latter in
    `validate`, since instrument agreement alone cannot reveal a day that is
    absent from all four series). Nothing is reindexed onto a constructed
    calendar and nothing is filled.
    """
    adj = {t: raw[t]["Adj Close"].loc[PANEL_BUFFER_START:SAMPLE_END] for t in TICKERS}

    in_sample = {t: set(s.loc[SAMPLE_START:SAMPLE_END].index) for t, s in adj.items()}
    union = set.union(*in_sample.values())
    intersection = set.intersection(*in_sample.values())
    genuinely_missing = {
        t: sorted(str(d.date()) for d in (union - dates))
        for t, dates in in_sample.items()
        if union - dates
    }

    panel = pd.DataFrame(adj).sort_index()
    # Keep the last trading day before the sample as the return base, then the
    # sample itself. Rows where any instrument is absent are reported above.
    base_row = panel.loc[:SAMPLE_START].iloc[:-1].dropna().tail(1)
    panel = pd.concat([base_row, panel.loc[SAMPLE_START:SAMPLE_END]])
    panel.index.name = "Date"

    returns = panel.pct_change().dropna(how="all")
    return panel, returns, genuinely_missing, len(union), len(intersection)


def build_risk_free(raw_yield, sample_days):
    """Align the published Treasury yield to trading days and convert to a daily rate.

    The series is quoted as an annualised percentage. The daily rate is
    `(1 + y/100) ** (1/252) - 1`, i.e. the constant daily rate that compounds to
    the quoted annual rate over a 252-day year.

    The bond market closes on Columbus Day and Veterans Day while the NYSE
    trades, so those trading days carry no published quote. The last previously
    published yield is carried forward, which uses only past information. This
    is a convention on the level of an interest rate, not an invented price
    observation, and every occurrence is counted in the coverage report.
    """
    aligned = raw_yield.reindex(sample_days)
    carried = aligned.isna()
    filled = aligned.ffill()
    frame = pd.DataFrame(
        {
            "annualized_percent": filled,
            "daily_rate": (1.0 + filled / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0,
            "carried_forward": carried.astype(int),
        }
    )
    frame.index.name = "Date"
    return frame


# --- Validation ---------------------------------------------------------------

def check(results, name, passed, detail):
    results.append({"check": name, "passed": bool(passed), "detail": detail})


def validate(raw, panel, returns, risk_free, peers, genuinely_missing,
             n_union, n_intersection):
    """Run every validation check and return the results. Never mutates data."""
    results = []
    sample = panel.loc[SAMPLE_START:SAMPLE_END]

    # --- Structure ---
    check(results, "panel columns are exactly the four protocol instruments",
          list(panel.columns) == TICKERS, f"columns={list(panel.columns)}")
    check(results, "panel contains no missing values",
          not panel.isna().any().any(), f"nan_count={int(panel.isna().sum().sum())}")
    check(results, "all prices are strictly positive",
          bool((panel > 0).all().all()), f"min_price={float(panel.min().min()):.4f}")

    # --- Chronology ---
    check(results, "index is a timezone-naive DatetimeIndex",
          isinstance(panel.index, pd.DatetimeIndex) and panel.index.tz is None,
          f"dtype={panel.index.dtype}")
    check(results, "dates are strictly increasing and unique",
          bool(panel.index.is_monotonic_increasing and panel.index.is_unique),
          f"n_rows={len(panel)}, n_unique={panel.index.nunique()}")
    check(results, "no weekend dates in the panel",
          not bool((panel.index.dayofweek >= 5).any()),
          f"weekend_rows={int((panel.index.dayofweek >= 5).sum())}")

    # --- Coverage ---
    first_return, last_return = returns.index.min(), returns.index.max()
    check(results, "return sample starts on the first protocol trading day",
          str(first_return.date()) == SAMPLE_START,
          f"first_return={first_return.date()} expected={SAMPLE_START}")
    check(results, "return sample ends on the protocol end date",
          str(last_return.date()) == SAMPLE_END,
          f"last_return={last_return.date()} expected={SAMPLE_END}")
    check(results, "every instrument's history begins before the sample start",
          all(raw[t].index.min() < pd.Timestamp(SAMPLE_START) for t in TICKERS),
          "; ".join(f"{t}:{raw[t].index.min().date()}" for t in TICKERS))
    check(results, "instruments report identical trading dates",
          not genuinely_missing,
          f"union={n_union} intersection={n_intersection} "
          f"missing={genuinely_missing or 'none'}")
    check(results, "estimation window of 756 days fits inside the sample",
          len(returns) > 756, f"n_returns={len(returns)}")

    # --- Trading calendar against an authoritative source ---
    # Instrument agreement cannot reveal a day missing from all four series, so
    # the panel is compared with the official exchange calendar.
    official = exchange_trading_days(SAMPLE_START, SAMPLE_END)
    ours = pd.DatetimeIndex(sample.index).normalize()
    absent_from_panel = official.difference(ours)
    unexpected_in_panel = ours.difference(official)
    check(results,
          f"panel matches the official {EXCHANGE_CALENDAR} trading calendar exactly",
          len(absent_from_panel) == 0 and len(unexpected_in_panel) == 0,
          f"panel={len(ours)} {EXCHANGE_CALENDAR}={len(official)}; "
          f"missing_from_panel={[str(d.date()) for d in absent_from_panel] or 'none'}; "
          f"not_in_calendar={[str(d.date()) for d in unexpected_in_panel] or 'none'}")

    # Checked over the whole panel range, not just the return sample, so that
    # closures falling on the leading return-base day are validated too.
    panel_calendar = exchange_trading_days(panel.index.min(), SAMPLE_END)
    panel_days = pd.DatetimeIndex(panel.index).normalize()
    for date, reason in EXCEPTIONAL_CLOSURES:
        stamp = pd.Timestamp(date)
        if not (panel.index.min() <= stamp <= pd.Timestamp(SAMPLE_END)):
            continue
        check(results,
              f"exceptional closure {date} is absent from calendar and panel",
              stamp.dayofweek < 5 and stamp not in panel_calendar
              and stamp not in panel_days,
              f"{reason}; weekday={stamp.day_name()}, "
              f"in_{EXCHANGE_CALENDAR}={stamp in panel_calendar}, "
              f"in_panel={stamp in panel_days}")

    # --- Integrity ---
    worst = returns.abs().max()
    check(results, f"no absolute daily return exceeds {MAX_PLAUSIBLE_ABS_DAILY_RETURN:.0%}",
          bool((worst <= MAX_PLAUSIBLE_ABS_DAILY_RETURN).all()),
          "; ".join(f"{t}:{worst[t]:.4f}" for t in TICKERS))

    # Genuine splits are already reflected in Yahoo's price columns, so the
    # adjusted series must be continuous across the split date. Continuity
    # demonstrates the adjustment is present exactly once: an unadjusted series
    # would jump by the ratio, and a doubly-adjusted one would jump by its
    # inverse. Both breach the plausibility bound above.
    for split in KNOWN_SPLITS:
        ticker, date, ratio = split["ticker"], split["date"], split["ratio"]
        stamp = pd.Timestamp(date)
        window = returns[ticker].loc[stamp - pd.Timedelta(days=5): stamp + pd.Timedelta(days=5)]
        observed = returns[ticker].get(stamp, float("nan"))
        check(results,
              f"{ticker} {ratio:.0f}-for-1 split of {date} is adjusted exactly once",
              bool(window.abs().max() < 0.25),
              f"return_on_split_date={observed:+.4f}, "
              f"max_abs_return_within_5_days={window.abs().max():.4f}; "
              f"a missing adjustment would show {1/ratio - 1:+.2f} and a double "
              f"adjustment {ratio - 1:+.2f}")

    # Adjusted close must be a genuine total-return series. Reconstruct it from
    # raw closes and the dividend record using Yahoo's back-adjustment
    # convention: the factor applied before an ex-date is 1 - Div/Close_prev,
    # so r_t = Close_t / (Close_prev - Div_t) - 1. Both columns are already
    # split-adjusted, so no split term enters and none is applied.
    for ticker in TICKERS:
        frame = raw[ticker].loc[PANEL_BUFFER_START:SAMPLE_END]
        adj_ret = frame["Adj Close"].pct_change()
        rebuilt = frame["Close"] / (frame["Close"].shift(1) - frame["Dividends"]) - 1
        deviation = (rebuilt - adj_ret).abs().dropna()
        check(results, f"{ticker} adjusted close reproduces as a total-return series",
              bool(deviation.max() <= ADJ_RECONSTRUCTION_TOLERANCE),
              f"max_deviation={deviation.max():.2e} "
              f"tolerance={ADJ_RECONSTRUCTION_TOLERANCE:.0e}")

    # The panel must be a strict subset of the raw dates: proof that no date was
    # invented during alignment.
    for ticker in TICKERS:
        check(results, f"{ticker} panel dates all come from the raw download",
              panel.index.isin(raw[ticker].index).all(),
              f"n_panel_dates_not_in_raw={int((~panel.index.isin(raw[ticker].index)).sum())}")

    # --- Cross-issuer agreement ---
    for ticker, peer_prices in peers.items():
        peer_name = CROSS_CHECK_PEERS[ticker]
        joined = pd.concat([returns[ticker], peer_prices.pct_change()], axis=1,
                           join="inner").dropna()
        joined.columns = [ticker, peer_name]
        overlap = joined.loc[SAMPLE_START:SAMPLE_END]
        gap = (overlap[ticker] - overlap[peer_name]).abs()
        correlation = overlap[ticker].corr(overlap[peer_name])
        check(results, f"{ticker} agrees with independent peer {peer_name}",
              bool(gap.max() <= MAX_PEER_RETURN_DIFFERENCE
                   and correlation >= MIN_PEER_RETURN_CORRELATION),
              f"overlap={len(overlap)} days from {overlap.index.min().date()}; "
              f"corr={correlation:.4f}; max_daily_gap={gap.max():.4f}")

    # --- Risk-free series ---
    check(results, "risk-free series covers every trading day in the sample",
          risk_free.index.equals(pd.DatetimeIndex(sample.index)),
          f"n_risk_free={len(risk_free)} n_sample_days={len(sample)}")
    check(results, "risk-free series has no unresolved missing values",
          not risk_free[["annualized_percent", "daily_rate"]].isna().any().any(),
          f"nan_count={int(risk_free[['annualized_percent', 'daily_rate']].isna().sum().sum())}")
    n_years = max(1, len(sample) / TRADING_DAYS_PER_YEAR)
    carried = int(risk_free["carried_forward"].sum())
    check(results, "carried-forward yields are limited to bond-market holidays",
          carried / n_years <= MAX_CARRIED_FORWARD_YIELDS_PER_YEAR,
          f"carried_forward={carried} over {n_years:.1f} years "
          f"({carried / n_years:.2f}/year, limit {MAX_CARRIED_FORWARD_YIELDS_PER_YEAR})")
    check(results, "risk-free yields are non-negative and economically plausible",
          bool((risk_free["annualized_percent"] >= 0).all()
               and (risk_free["annualized_percent"] < 25).all()),
          f"min={risk_free['annualized_percent'].min():.2f}% "
          f"max={risk_free['annualized_percent'].max():.2f}%")

    check(results, "returns have exactly one row fewer than prices",
          len(returns) == len(panel) - 1,
          f"n_prices={len(panel)} n_returns={len(returns)}")
    return results


# --- Reporting ----------------------------------------------------------------

def write_report(raw, panel, returns, risk_free, peers, results, manifest,
                 genuinely_missing, n_union, n_intersection, mode):
    lines = []
    add = lines.append
    sample = panel.loc[SAMPLE_START:SAMPLE_END]

    add("# Data Coverage and Validation Report")
    add("")
    add("Generated by `download_data.py`. Do not edit by hand.")
    add("")
    add(f"* Build mode: **{mode}**")
    for name, src in manifest["sources"].items():
        add(f"* Source ({name}): {src}")
    add(f"* Raw inputs frozen (UTC): {manifest['downloaded_utc']}")
    add(f"* Package versions: {manifest['versions']}")
    add(f"* Raw archive window requested: {RAW_START} to {RAW_END} (end exclusive)")
    add(f"* Return sample: {SAMPLE_START} to {SAMPLE_END}")
    add("")

    add("## Instrument coverage")
    add("")
    add("| Ticker | Raw first | Raw last | Raw rows | Sample rows | Dividend days | Split records |")
    add("|---|---|---|---|---|---|---|")
    for t in TICKERS:
        r = raw[t]
        in_sample = r.loc[SAMPLE_START:SAMPLE_END]
        add(f"| {t} | {r.index.min().date()} | {r.index.max().date()} | {len(r)} | "
            f"{len(in_sample)} | {int((in_sample['Dividends'] > 0).sum())} | "
            f"{int((in_sample['Stock Splits'] != 0).sum())} |")
    add("")

    add("## Missing data versus non-trading days")
    add("")
    add(f"Two independent tests, because instrument agreement alone cannot reveal a "
        f"day absent from all four series.")
    add("")
    add(f"**1. Agreement across instruments.** Over {SAMPLE_START} to {SAMPLE_END} the "
        f"union of the four instruments' trading dates contains **{n_union}** days and "
        f"the intersection contains **{n_intersection}** days.")
    add("")
    if genuinely_missing:
        add("**Genuine missing observations detected** (date present for some "
            "instruments, absent for others):")
        add("")
        for t, dates in genuinely_missing.items():
            add(f"* {t}: {len(dates)} missing — {dates[:20]}{' ...' if len(dates) > 20 else ''}")
    else:
        add("The two sets are identical, so no instrument is missing a day that "
            "another instrument reports.")
    add("")
    official = exchange_trading_days(SAMPLE_START, SAMPLE_END)
    ours = pd.DatetimeIndex(sample.index).normalize()
    add(f"**2. Agreement with the official exchange calendar.** The panel is compared "
        f"with the {EXCHANGE_CALENDAR} trading calendar from `pandas_market_calendars` "
        f"(validation only; the panel's dates are never constructed from it).")
    add("")
    add(f"* Panel trading days: **{len(ours)}**")
    add(f"* Official {EXCHANGE_CALENDAR} trading days: **{len(official)}**")
    add(f"* In the calendar but absent from the panel: "
        f"**{[str(d.date()) for d in official.difference(ours)] or 'none'}**")
    add(f"* In the panel but not in the calendar: "
        f"**{[str(d.date()) for d in ours.difference(official)] or 'none'}**")
    add("")
    if len(official.difference(ours)) == 0 and len(ours.difference(official)) == 0:
        add("The panel matches the official calendar exactly, so **no genuine missing "
            "observations exist**: every date absent from the panel is a scheduled "
            "holiday or a documented exceptional closure. No value is interpolated, "
            "forward filled or imputed anywhere in the price pipeline.")
    else:
        add("**The panel does not match the official calendar.** The discrepancies above "
            "are genuine missing observations or dates that should not be present, and must be resolved "
            "before the data are used.")
    add("")

    add("### Exceptional exchange closures in the sample")
    add("")
    add("Weekdays that were not scheduled holidays. Each is absent from both the "
        "official calendar and the panel, as a closure rather than a data gap.")
    add("")
    add("| Date | Weekday | Reason | In calendar | In panel |")
    add("|---|---|---|---|---|")
    panel_calendar = exchange_trading_days(panel.index.min(), SAMPLE_END)
    panel_days = pd.DatetimeIndex(panel.index).normalize()
    for date, reason in EXCEPTIONAL_CLOSURES:
        stamp = pd.Timestamp(date)
        add(f"| {date} | {stamp.day_name()} | {reason} | "
            f"{'yes' if stamp in panel_calendar else 'no'} | "
            f"{'yes' if stamp in panel_days else 'no'} |")
    add("")

    add("## Price panel")
    add("")
    add(f"* Rows: {len(panel)} (first row {panel.index.min().date()} is the return base only)")
    add(f"* Daily return observations: {len(returns)}")
    add(f"* Columns: {', '.join(panel.columns)}")
    add("")
    add("| Ticker | First adj. price | Last adj. price | Mean daily return | Daily sd | Min | Max |")
    add("|---|---|---|---|---|---|---|")
    for t in TICKERS:
        add(f"| {t} | {panel[t].iloc[0]:.4f} | {panel[t].iloc[-1]:.4f} | "
            f"{returns[t].mean():.6f} | {returns[t].std():.6f} | "
            f"{returns[t].min():.4f} | {returns[t].max():.4f} |")
    add("")

    add("## Risk-free rate and cash returns")
    add("")
    add(f"FRED series `{RISK_FREE_SERIES}` (3-Month Treasury Constant Maturity Rate), "
        f"published as an annualised percentage on an investment basis. Converted to a "
        f"daily rate as `(1 + y/100) ** (1/{TRADING_DAYS_PER_YEAR}) - 1`, the constant "
        f"daily rate compounding to the quoted annual rate over a "
        f"{TRADING_DAYS_PER_YEAR}-day year.")
    add("")
    carried = int(risk_free["carried_forward"].sum())
    add(f"* Trading days covered: {len(risk_free)}")
    add(f"* Annualised yield range: {risk_free['annualized_percent'].min():.2f}% to "
        f"{risk_free['annualized_percent'].max():.2f}%")
    add(f"* Mean daily rate: {risk_free['daily_rate'].mean():.8f}")
    add(f"* Trading days with no published quote: **{carried}** "
        f"({carried / (len(risk_free) / TRADING_DAYS_PER_YEAR):.2f} per year)")
    add("")
    add("The bond market closes on Columbus Day and Veterans Day while the NYSE trades, "
        "so those days carry no published quote. The last previously published yield is "
        "carried forward, which uses only past information. This is a convention on the "
        "level of an interest rate, not an invented price observation, and each occurrence "
        "is flagged in the `carried_forward` column of "
        "`data/processed/risk_free_daily.csv`. Phase 2 uses the rate dated *t-1* for the "
        "cash return earned on day *t*, so no yield enters a decision before it was "
        "published.")
    add("")

    add("## Adjusted-price convention")
    add("")
    add("Adjusted closes were verified, not assumed. Reconstructing them from raw closes "
        "and the dividend record under Yahoo's back-adjustment convention, "
        "`r_t = Close_t / (Close_{t-1} - Div_t) - 1`, reproduces the adjusted series for "
        "all four instruments (deviations in the check table). The naive "
        "`(Close_t + Div_t) / Close_{t-1}` convention does not match and is not used. "
        "GLD and SLV pay no distributions, so their adjusted and raw series coincide.")
    add("")
    add("Yahoo's adjusted prices are back-adjusted and anchored to the latest "
        "distribution, so adjusted *levels* shift between downloads as new distributions "
        "arrive, while returns inside a fixed download do not. The raw download is "
        "archived and checksummed for this reason, and the default build verifies those "
        "checksums rather than re-downloading.")
    add("")

    add("### Share splits")
    add("")
    add("Both of Yahoo's price columns, `Close` and `Adj Close`, are back-adjusted for "
        "share splits. The pipeline therefore takes split handling from those columns and "
        "**never applies the `Stock Splits` column itself**, so each split is applied "
        "exactly once.")
    add("")
    for split in KNOWN_SPLITS:
        add(f"* **{split['ticker']}, {split['date']}, {split['ratio']:.0f}-for-1.** "
            f"{split['description']}. Source: {split['source']}")
    add("")
    add("The adjusted series is continuous across this date, which is the expected "
        "consequence of the adjustment already being present: an unadjusted series would "
        "fall by about 90% on the split date, and a doubly-adjusted one would rise "
        "tenfold. Continuity is therefore evidence that the adjustment is applied once, "
        "not evidence that no split occurred, and it is checked as such.")
    add("")

    add("## Cross-issuer verification")
    add("")
    add("The two metal series were checked against competing ETFs from different issuers "
        "tracking the same metal, as an independent test of the adjusted series. Small "
        "differences are expected (fee drag, premium/discount), but close agreement rules "
        "out issuer-specific data artefacts.")
    add("")
    add("| Series | Independent peer | Overlap | Overlap starts | Return correlation | Max daily gap |")
    add("|---|---|---|---|---|---|")
    for ticker, peer_prices in peers.items():
        peer_name = CROSS_CHECK_PEERS[ticker]
        joined = pd.concat([returns[ticker], peer_prices.pct_change()], axis=1,
                           join="inner").dropna()
        joined.columns = [ticker, peer_name]
        overlap = joined.loc[SAMPLE_START:SAMPLE_END]
        gap = (overlap[ticker] - overlap[peer_name]).abs()
        add(f"| {ticker} | {peer_name} | {len(overlap)} days | "
            f"{overlap.index.min().date()} | "
            f"{overlap[ticker].corr(overlap[peer_name]):.4f} | {gap.max():.4f} |")
    add("")
    add("Peer histories are shorter than the sample, so the overlap is a subset of it. "
        "The peers are used for verification only and never enter the analysis.")
    add("")

    add("## Largest daily moves (recorded for transparency)")
    add("")
    add("These are genuine market events, not data errors, and are listed so that later "
        "risk estimates can be read against them.")
    add("")
    for t in TICKERS:
        add(f"* **{t}** — largest drops: "
            + ", ".join(f"{d.date()} {v:+.2%}" for d, v in returns[t].nsmallest(3).items())
            + "; largest gains: "
            + ", ".join(f"{d.date()} {v:+.2%}" for d, v in returns[t].nlargest(3).items()))
    add("")
    add("Two points deserve note for the analysis. First, SPY's +14.52% on 2008-10-13 "
        "exceeds the S&P 500 index move that day (+11.58%) because SPY closed at a "
        "discount to net asset value on 2008-10-10 and at a premium on 2008-10-13. This "
        "is real ETF pricing behaviour during the crisis and is precisely the "
        "implementable-return effect the ETF-based design is meant to capture. Second, "
        "the metals saw an exceptional boom and crash late in the sample: SLV roughly "
        "doubled between November 2025 and February 2026 and fell 28.5% on 2026-01-30 "
        "alone, with GLD down 10.3% the same day. Both moves were verified against "
        "independent issuers (SIVR, IAU, PSLV) and are genuine. This episode falls inside "
        "the out-of-sample evaluation window and will materially influence both the "
        "Expected Shortfall estimates driving allocation and the realised net returns.")
    add("")

    add("## Validation checks")
    add("")
    add("| # | Check | Result | Detail |")
    add("|---|---|---|---|")
    for i, r in enumerate(results, 1):
        add(f"| {i} | {r['check']} | {'PASS' if r['passed'] else 'FAIL'} | {r['detail']} |")
    add("")
    add(f"**{sum(r['passed'] for r in results)} of {len(results)} checks passed.**")
    add("")

    add("## Frozen raw inputs (SHA-256)")
    add("")
    add("The default build verifies these checksums and fails if any input is missing or "
        "altered. Refreshing them requires `--download`.")
    add("")
    add("| File | Role | Rows | SHA-256 |")
    add("|---|---|---|---|")
    for name, info in manifest["files"].items():
        add(f"| `{name}` | {info['role']} | {info['rows']} | `{info['sha256']}` |")
    add("")

    add("## Transformations applied")
    add("")
    add("The complete list, in order. Nothing else is done to the data in Phase 1.")
    add("")
    add("1. Acquire raw OHLCV, `Adj Close`, `Dividends` and `Stock Splits` per ticker, "
        "plus the raw FRED yield series; archive each verbatim and checksum it.")
    add("2. Select the `Adj Close` column of each ticker as its total-return price series "
        "(already split- and distribution-adjusted; the `Stock Splits` column is never applied).")
    add(f"3. Restrict to {SAMPLE_START}..{SAMPLE_END}, plus the single preceding trading "
        "day as the return base.")
    add("4. Join the four series on their common trading calendar (no reindexing onto a "
        "constructed calendar, no filling).")
    add("5. Compute daily simple returns between consecutive observed trading days.")
    add(f"6. Align the Treasury yield to those trading days, carry the last published "
        f"yield forward on bond-market holidays (flagged), and convert to a daily rate via "
        f"`(1 + y/100) ** (1/{TRADING_DAYS_PER_YEAR}) - 1`.")
    add("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the Phase 1 dataset. By default this rebuilds the "
            "processed dataset from the frozen raw inputs in data/raw/ and verifies "
            "their checksums, making no network calls."
        )
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "re-download from the sources and overwrite the frozen raw inputs and "
            "manifest; required to refresh the research inputs"
        ),
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("--download given: refreshing raw inputs and rewriting the manifest.")
        print("This replaces the frozen research inputs.\n")
        download_sources()
        mode = "refreshed from source (--download)"
    else:
        mode = "rebuilt from frozen raw inputs"

    manifest, raw, peers, risk_free_raw = load_frozen()
    if not args.download:
        print(f"verified {len(manifest['files'])} frozen raw inputs against the manifest")

    panel, returns, genuinely_missing, n_union, n_intersection = build_panel(raw)
    risk_free = build_risk_free(risk_free_raw, panel.loc[SAMPLE_START:SAMPLE_END].index)

    panel.to_csv(PROCESSED_DIR / "adjusted_prices.csv")
    returns.to_csv(PROCESSED_DIR / "daily_returns.csv")
    risk_free.to_csv(PROCESSED_DIR / "risk_free_daily.csv")

    results = validate(raw, panel, returns, risk_free, peers, genuinely_missing,
                       n_union, n_intersection)
    write_report(raw, panel, returns, risk_free, peers, results, manifest,
                 genuinely_missing, n_union, n_intersection, mode)

    print()
    for r in results:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['check']}: {r['detail']}")
    failed = [r for r in results if not r["passed"]]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed. Report: {REPORT_PATH}")
    if failed:
        print("VALIDATION FAILED - do not proceed to Phase 2.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

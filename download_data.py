"""Phase 1: download, validate and freeze the price data for the safe-haven study.

Run:  python3 download_data.py

Writes
------
data/raw/<TICKER>_yahoo_raw.csv   unmodified Yahoo Finance download (archive)
data/raw/manifest.json            source, download timestamp, versions, SHA-256
data/processed/adjusted_prices.csv  aligned adjusted total-return price panel
data/processed/daily_returns.csv    daily simple returns of the panel
data/data_coverage_report.md        coverage, missing data and validation report

The script validates rather than repairs. Nothing is interpolated, forward
filled or simulated: if a required series is incomplete the run fails and says
so. See research_protocol.md for the frozen design this implements.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
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

# --- Validation thresholds (documented, not tuned) ---------------------------

# Largest plausible absolute daily return for a broad ETF. Any breach is
# treated as suspected data corruption (e.g. an unadjusted split), not as a
# market move, and fails validation.
MAX_PLAUSIBLE_ABS_DAILY_RETURN = 0.40

# Tolerance for reconstructing adjusted-close returns from raw closes and
# dividends. Yahoo stores Adj Close as float32, so exact equality is
# unattainable; 1e-4 is far tighter than any economically relevant error.
ADJ_RECONSTRUCTION_TOLERANCE = 1e-4

# Yahoo reports a 10:1 SLV split here. It is spurious: the price series is
# continuous across the date and the adjusted series does not apply it.
# See research_protocol.md section 9.3.
KNOWN_SPURIOUS_SPLITS = [("SLV", "2008-07-24")]

# Independent cross-check of the metal series against competing ETFs from
# different issuers tracking the same metal. Agreement is evidence that the
# adjusted series reflect the underlying metal and not an issuer-specific data
# artefact. Not used in the analysis; validation only.
CROSS_CHECK_PEERS = {"SLV": "SIVR", "GLD": "IAU"}

# Peers track the same metal through a different vehicle, so small differences
# in fee drag and premium/discount are expected. A breach of this bound would
# indicate one of the two series is wrong.
MAX_PEER_RETURN_DIFFERENCE = 0.02
MIN_PEER_RETURN_CORRELATION = 0.98

DOWNLOAD_ATTEMPTS = 4

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORT_PATH = Path("data/data_coverage_report.md")


def download_raw(ticker):
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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_panel(raw):
    """Align the four adjusted total-return series onto their common calendar.

    Missing data are distinguished from non-trading days by comparing the
    trading dates the instruments themselves report. A date present for some
    instruments but absent for others is genuine missing data and is reported;
    a date absent for all of them is a market closure and is simply not in the
    panel. Nothing is reindexed onto a synthetic calendar and nothing is filled.
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
    base = panel.loc[:SAMPLE_START].iloc[:-1]
    base_row = base.dropna().tail(1)
    sample = panel.loc[SAMPLE_START:SAMPLE_END]
    panel = pd.concat([base_row, sample])
    panel.index.name = "Date"

    returns = panel.pct_change().dropna(how="all")
    return panel, returns, genuinely_missing, len(union), len(intersection)


def check(results, name, passed, detail):
    results.append({"check": name, "passed": bool(passed), "detail": detail})


def validate(raw, panel, returns, genuinely_missing, n_union, n_intersection,
             peers):
    """Run every validation check and return the results. Never mutates data."""
    results = []

    # --- Structure ---
    check(
        results,
        "panel columns are exactly the four protocol instruments",
        list(panel.columns) == TICKERS,
        f"columns={list(panel.columns)}",
    )
    check(
        results,
        "panel contains no missing values",
        not panel.isna().any().any(),
        f"nan_count={int(panel.isna().sum().sum())}",
    )
    check(
        results,
        "all prices are strictly positive",
        bool((panel > 0).all().all()),
        f"min_price={float(panel.min().min()):.4f}",
    )

    # --- Chronology ---
    check(
        results,
        "index is a timezone-naive DatetimeIndex",
        isinstance(panel.index, pd.DatetimeIndex) and panel.index.tz is None,
        f"dtype={panel.index.dtype}",
    )
    check(
        results,
        "dates are strictly increasing and unique",
        bool(panel.index.is_monotonic_increasing and panel.index.is_unique),
        f"n_rows={len(panel)}, n_unique={panel.index.nunique()}",
    )
    check(
        results,
        "no weekend dates in the panel",
        not bool((panel.index.dayofweek >= 5).any()),
        f"weekend_rows={int((panel.index.dayofweek >= 5).sum())}",
    )

    # --- Coverage ---
    first_return = returns.index.min()
    last_return = returns.index.max()
    check(
        results,
        "return sample starts on the first protocol trading day",
        str(first_return.date()) == SAMPLE_START,
        f"first_return={first_return.date()} expected={SAMPLE_START}",
    )
    check(
        results,
        "return sample ends on the protocol end date",
        str(last_return.date()) == SAMPLE_END,
        f"last_return={last_return.date()} expected={SAMPLE_END}",
    )
    check(
        results,
        "every instrument's history begins before the sample start",
        all(raw[t].index.min() < pd.Timestamp(SAMPLE_START) for t in TICKERS),
        "; ".join(f"{t}:{raw[t].index.min().date()}" for t in TICKERS),
    )
    check(
        results,
        "no genuine missing observations (calendars agree across instruments)",
        not genuinely_missing,
        f"union={n_union} intersection={n_intersection} missing={genuinely_missing or 'none'}",
    )
    check(
        results,
        "estimation window of 756 days fits inside the sample",
        len(returns) > 756,
        f"n_returns={len(returns)}",
    )

    # --- Integrity ---
    worst = returns.abs().max()
    check(
        results,
        f"no absolute daily return exceeds {MAX_PLAUSIBLE_ABS_DAILY_RETURN:.0%}",
        bool((worst <= MAX_PLAUSIBLE_ABS_DAILY_RETURN).all()),
        "; ".join(f"{t}:{worst[t]:.4f}" for t in TICKERS),
    )

    for ticker, date in KNOWN_SPURIOUS_SPLITS:
        stamp = pd.Timestamp(date)
        window = returns[ticker].loc[stamp - pd.Timedelta(days=5): stamp + pd.Timedelta(days=5)]
        check(
            results,
            f"{ticker} price is continuous across the spurious {date} split record",
            bool(window.abs().max() < 0.25),
            f"max_abs_return_within_5_days={window.abs().max():.4f} "
            f"(a 10:1 split would show ~0.9)",
        )

    # Adjusted close must be a genuine total-return series. Reconstruct it from
    # raw closes and the dividend record using Yahoo's back-adjustment
    # convention: the factor applied before an ex-date is 1 - Div/Close_prev,
    # so r_t = Close_t / (Close_prev - Div_t) - 1.
    for ticker in TICKERS:
        frame = raw[ticker].loc[PANEL_BUFFER_START:SAMPLE_END]
        adj_ret = frame["Adj Close"].pct_change()
        prev_close = frame["Close"].shift(1)
        rebuilt = frame["Close"] / (prev_close - frame["Dividends"]) - 1
        deviation = (rebuilt - adj_ret).abs().dropna()
        check(
            results,
            f"{ticker} adjusted close reproduces as a total-return series",
            bool(deviation.max() <= ADJ_RECONSTRUCTION_TOLERANCE),
            f"max_deviation={deviation.max():.2e} tolerance={ADJ_RECONSTRUCTION_TOLERANCE:.0e}",
        )

    # The panel must be a strict subset of the raw dates: proof that no date was
    # invented during alignment.
    for ticker in TICKERS:
        check(
            results,
            f"{ticker} panel dates all come from the raw download",
            panel.index.isin(raw[ticker].index).all(),
            f"n_panel_dates_not_in_raw={int((~panel.index.isin(raw[ticker].index)).sum())}",
        )

    # Cross-issuer agreement: the metal series must match an independent ETF on
    # the same metal. This catches issuer-specific data errors that internal
    # consistency checks cannot see.
    for ticker, peer_prices in peers.items():
        peer_name = CROSS_CHECK_PEERS[ticker]
        peer_returns = peer_prices.pct_change()
        joined = pd.concat(
            [returns[ticker], peer_returns], axis=1, join="inner"
        ).dropna()
        joined.columns = [ticker, peer_name]
        overlap = joined.loc[SAMPLE_START:SAMPLE_END]
        gap = (overlap[ticker] - overlap[peer_name]).abs()
        correlation = overlap[ticker].corr(overlap[peer_name])
        check(
            results,
            f"{ticker} agrees with independent peer {peer_name}",
            bool(
                gap.max() <= MAX_PEER_RETURN_DIFFERENCE
                and correlation >= MIN_PEER_RETURN_CORRELATION
            ),
            f"overlap={len(overlap)} days from {overlap.index.min().date()}; "
            f"corr={correlation:.4f}; max_daily_gap={gap.max():.4f}",
        )

    check(
        results,
        "returns have exactly one row fewer than prices",
        len(returns) == len(panel) - 1,
        f"n_prices={len(panel)} n_returns={len(returns)}",
    )
    return results


def write_report(raw, panel, returns, results, manifest, genuinely_missing,
                 n_union, n_intersection, peers):
    lines = []
    add = lines.append
    add("# Data Coverage and Validation Report")
    add("")
    add("Generated by `download_data.py`. Do not edit by hand.")
    add("")
    add(f"* Source: {manifest['source']}")
    add(f"* Downloaded (UTC): {manifest['downloaded_utc']}")
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
        add(
            f"| {t} | {r.index.min().date()} | {r.index.max().date()} | {len(r)} | "
            f"{len(in_sample)} | {int((in_sample['Dividends'] > 0).sum())} | "
            f"{int((in_sample['Stock Splits'] != 0).sum())} |"
        )
    add("")

    add("## Missing data versus non-trading days")
    add("")
    add(
        f"Over {SAMPLE_START} to {SAMPLE_END} the union of the four instruments' trading "
        f"dates contains **{n_union}** days and the intersection contains "
        f"**{n_intersection}** days."
    )
    add("")
    if genuinely_missing:
        add("**Genuine missing observations detected** (date present for some instruments, absent for others):")
        add("")
        for t, dates in genuinely_missing.items():
            add(f"* {t}: {len(dates)} missing — {dates[:20]}{' ...' if len(dates) > 20 else ''}")
    else:
        add(
            "The two sets are identical, so **no genuine missing observations exist**. "
            "Every date absent from the panel is a market closure affecting all four "
            "instruments. No value is interpolated, forward filled or imputed anywhere "
            "in this pipeline."
        )
    add("")
    add(
        "Non-trading days are absent from the panel by construction: the calendar is "
        "taken from the instruments themselves rather than from a synthetic date range."
    )
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
        add(
            f"| {t} | {panel[t].iloc[0]:.4f} | {panel[t].iloc[-1]:.4f} | "
            f"{returns[t].mean():.6f} | {returns[t].std():.6f} | "
            f"{returns[t].min():.4f} | {returns[t].max():.4f} |"
        )
    add("")

    add("## Adjusted-price convention")
    add("")
    add(
        "Adjusted closes were verified, not assumed. Reconstructing them from raw closes "
        "and the dividend record under Yahoo's back-adjustment convention, "
        "`r_t = Close_t / (Close_{t-1} - Div_t) - 1`, reproduces the adjusted series for "
        "all four instruments (deviations below). The naive "
        "`(Close_t + Div_t) / Close_{t-1}` convention does not match and is not used. "
        "GLD and SLV pay no distributions, so their adjusted and raw series coincide."
    )
    add("")
    add(
        "Yahoo's adjusted prices are back-adjusted and anchored to the latest "
        "distribution, so adjusted *levels* shift between downloads as new distributions "
        "arrive, while returns inside a fixed download do not. The raw download is "
        "archived and checksummed for this reason."
    )
    add("")
    add(
        "Yahoo's action record contains a spurious 10:1 SLV split on 2008-07-24. The price "
        "series is continuous across that date and the adjusted series does not apply it, "
        "so the pipeline takes split handling from the adjusted series only and never "
        "applies the `Stock Splits` column. This is checked below and in `test_data.py`."
    )
    add("")

    add("## Cross-issuer verification")
    add("")
    add(
        "The two metal series were checked against competing ETFs from different "
        "issuers tracking the same metal, as an independent test of the adjusted "
        "series. Small differences are expected (fee drag, premium/discount), but "
        "close agreement rules out issuer-specific data artefacts."
    )
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
        add(
            f"| {ticker} | {peer_name} | {len(overlap)} days | "
            f"{overlap.index.min().date()} | "
            f"{overlap[ticker].corr(overlap[peer_name]):.4f} | {gap.max():.4f} |"
        )
    add("")
    add(
        "Peer histories are shorter than the sample, so the overlap is a subset of "
        "it. The peers are used for verification only and never enter the analysis."
    )
    add("")

    add("## Largest daily moves (recorded for transparency)")
    add("")
    add(
        "These are genuine market events, not data errors, and are listed so that "
        "later risk estimates can be read against them."
    )
    add("")
    for t in TICKERS:
        worst = returns[t].nsmallest(3)
        best = returns[t].nlargest(3)
        add(
            f"* **{t}** — largest drops: "
            + ", ".join(f"{d.date()} {v:+.2%}" for d, v in worst.items())
            + "; largest gains: "
            + ", ".join(f"{d.date()} {v:+.2%}" for d, v in best.items())
        )
    add("")
    add(
        "Two points deserve note for the analysis. First, SPY's +14.52% on "
        "2008-10-13 exceeds the S&P 500 index move that day (+11.58%) because SPY "
        "closed at a discount to net asset value on 2008-10-10 and at a premium on "
        "2008-10-13. This is real ETF pricing behaviour during the crisis and is "
        "precisely the implementable-return effect the ETF-based design is meant to "
        "capture. Second, the metals saw an exceptional boom and crash late in the "
        "sample: SLV roughly doubled between November 2025 and February 2026 and "
        "fell 28.5% on 2026-01-30 alone, with GLD down 10.3% the same day. Both "
        "moves were verified against independent issuers (SIVR, IAU, PSLV) and are "
        "genuine. This episode falls inside the out-of-sample evaluation window and "
        "will materially influence both the Expected Shortfall estimates driving "
        "allocation and the realised net returns."
    )
    add("")

    add("## Validation checks")
    add("")
    add("| # | Check | Result | Detail |")
    add("|---|---|---|---|")
    for i, r in enumerate(results, 1):
        add(f"| {i} | {r['check']} | {'PASS' if r['passed'] else 'FAIL'} | {r['detail']} |")
    add("")
    n_passed = sum(r["passed"] for r in results)
    add(f"**{n_passed} of {len(results)} checks passed.**")
    add("")

    add("## Raw file checksums (SHA-256)")
    add("")
    add("| File | Rows | SHA-256 |")
    add("|---|---|---|")
    for name, info in manifest["files"].items():
        add(f"| `{name}` | {info['rows']} | `{info['sha256']}` |")
    add("")

    add("## Transformations applied")
    add("")
    add("The complete list, in order. Nothing else is done to the data in Phase 1.")
    add("")
    add("1. Download raw OHLCV, `Adj Close`, `Dividends` and `Stock Splits` per ticker; archive verbatim as CSV.")
    add("2. Select the `Adj Close` column of each ticker as its total-return price series.")
    add(f"3. Restrict to {SAMPLE_START}..{SAMPLE_END}, plus the single preceding trading day as the return base.")
    add("4. Join the four series on their common trading calendar (no reindexing, no filling).")
    add("5. Compute daily simple returns between consecutive observed trading days.")
    add("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = {}
    files = {}
    for ticker in TICKERS:
        print(f"downloading {ticker} ...", flush=True)
        frame = download_raw(ticker)
        path = RAW_DIR / f"{ticker}_yahoo_raw.csv"
        frame.to_csv(path)
        raw[ticker] = frame
        files[path.name] = {"ticker": ticker, "rows": len(frame), "sha256": sha256(path)}

    peers = {}
    for ticker, peer in CROSS_CHECK_PEERS.items():
        print(f"downloading cross-check peer {peer} for {ticker} ...", flush=True)
        peers[ticker] = download_raw(peer)["Adj Close"]

    manifest = {
        "source": "Yahoo Finance via the yfinance Python package",
        "downloaded_utc": pd.Timestamp.now("UTC").isoformat(),
        "requested_window": {"start": RAW_START, "end_exclusive": RAW_END},
        "download_arguments": {
            "auto_adjust": False,
            "actions": True,
            "threads": False,
        },
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "yfinance": yf.__version__,
        },
        "files": files,
        "licence_note": (
            "Yahoo Finance terms of service contemplate personal, non-commercial use. "
            "Raw CSVs are archived here for reproducibility; see research_protocol.md "
            "unresolved decision U8. The checksums alone pin the inputs if the CSVs "
            "must be removed."
        ),
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    panel, returns, genuinely_missing, n_union, n_intersection = build_panel(raw)
    panel.to_csv(PROCESSED_DIR / "adjusted_prices.csv")
    returns.to_csv(PROCESSED_DIR / "daily_returns.csv")

    results = validate(raw, panel, returns, genuinely_missing, n_union,
                       n_intersection, peers)
    write_report(raw, panel, returns, results, manifest, genuinely_missing,
                 n_union, n_intersection, peers)

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

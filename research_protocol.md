# Research Protocol (Frozen)

**Paper:** Does Dynamic Safe-Haven Allocation Improve Net Portfolio Returns?
**Conference deadline:** 25 October 2026
**Protocol frozen:** 20 October 2026, prior to any performance estimation.
**Status:** Phase 1 (data acquisition and validation) implemented. Phases 2 and 3 not started.

This document fixes the research design *before* out-of-sample results are
computed. It is the reference against which later code is checked. Changes to
Sections 1-8 after this freeze must be recorded in Section 10 with a reason
that is independent of the results obtained.

---

## 1. Research question

Does dynamically allocating between gold, silver and no metal improve the
**out-of-sample net return** of a conventional 60/40 US stock-bond portfolio?

The question is about *net* (after transaction cost) returns realised out of
sample, not about in-sample risk reduction.

## 2. Instruments

| Ticker | Instrument | Role |
|---|---|---|
| SPY | SPDR S&P 500 ETF Trust | US equity |
| AGG | iShares Core US Aggregate Bond ETF | US investment-grade bonds |
| GLD | SPDR Gold Shares | Gold |
| SLV | iShares Silver Trust | Silver |

Exchange-traded funds are used deliberately: they are investable, carry
observable prices, and embed real holding costs, so the estimated net returns
correspond to an implementable strategy rather than to a spot index.

## 3. Sample period

* **Return sample:** 2007-01-03 through 2026-09-30 (daily).
* **Price panel:** 2006-12-29 through 2026-09-30. The first date supplies the
  base price for the 2007-01-03 return only and is not itself an observation.
* **Out-of-sample evaluation period:** begins at the first monthly decision
  date at which a full 756-trading-day estimation window of in-sample data is
  available, i.e. the first rebalance date on or after the 756th trading day
  of the return sample (≈ January 2010), and ends 2026-09-30. See
  unresolved decision **U1**.

Validation outcome: all four instruments trade continuously over the whole
window. SLV, the shortest series, has an inception date of 2006-04-28, which
precedes the sample start, so **no truncation of the sample period is
required**. The sample period stated in the approved protocol is retained in
full.

Note that 2007-01-02 was not a trading day (NYSE closed for the national day
of mourning for President Ford). This is a genuine market closure, not a
missing observation.

## 4. Prices and returns

* Daily **adjusted total-return** prices, taken from the Yahoo Finance
  `Adj Close` field, which is back-adjusted for distributions and splits and
  therefore represents a dividend-reinvested total-return index.
* Daily simple returns: `r_t = P_t / P_{t-1} - 1`, computed only between
  consecutive observed trading days.
* No currency conversion (all instruments are USD-denominated).
* Reinvestment is implicitly continuous and at the closing price, which is the
  convention embedded in the adjusted series. See **U10**.

The adjusted-price convention was verified empirically rather than assumed;
see `data/data_coverage_report.md` and Section 9.

## 5. Candidate portfolios

Fixed target weights, identical across all decision dates:

| Candidate | SPY | AGG | GLD | SLV |
|---|---|---|---|---|
| `no_metal` | 0.60 | 0.40 | - | - |
| `gold` | 0.57 | 0.38 | 0.05 | - |
| `silver` | 0.57 | 0.38 | - | 0.05 |

The 5% metal sleeve is funded pro rata from equity and bonds, preserving the
60:40 equity-to-bond ratio within the non-metal part of each candidate.

## 6. Allocation rule

* **Estimation window:** rolling 756 trading days (≈ 3 years).
* **Rebalancing / decision frequency:** monthly.
* At each monthly decision date, construct the historical daily return series
  of each of the three candidates over the trailing 756 trading days, compute
  each candidate's **historical 95% Expected Shortfall (ES)**, and select the
  candidate with the **lowest** ES.
* ES is the mean of the worst 5% of daily returns in the window (historical,
  non-parametric; no distributional assumption).
* Only information available at the decision date is used. The selection rule
  never sees returns realised after the decision date, and in particular
  never uses future realised performance of the candidates.

## 7. Transaction costs

* **Principal assumption:** 10 basis points per traded dollar.
* **Robustness:** 25 and 50 basis points.
* Costs are charged on the absolute value of the change in each position at
  each rebalance, and are applied to the dynamic strategy **and** to every
  benchmark on the same basis, so that comparisons are like-for-like. See
  **U6** and **U9**.

## 8. Benchmarks and outcomes

**Benchmarks**

1. Traditional 60/40 (SPY/AGG), monthly rebalanced to target.
2. Static gold: 57/38/5 SPY/AGG/GLD, monthly rebalanced to target.
3. Static silver: 57/38/5 SPY/AGG/SLV, monthly rebalanced to target.
4. Conventional volatility-managed 60/40: scales exposure to a
   **prespecified** annualised volatility target, **no leverage** (weight on
   the risky portfolio capped at 1.0), residual held in cash. The target and
   the volatility estimator must be fixed before estimation; see **U4**.

**Outcomes**

* *Primary:* out-of-sample **net portfolio return**.
* *Secondary:* Sharpe ratio, 95% Expected Shortfall, maximum drawdown,
  portfolio turnover.

**Inference**

Paired out-of-sample net return differences (dynamic strategy minus each
benchmark), evaluated with a reproducible time-series **block bootstrap** with
a fixed random seed. Block structure and replication count in **U7**.

---

## 9. Data provenance and verification

**Source:** Yahoo Finance, accessed through the `yfinance` Python package.
Download date, package versions, row counts, coverage and SHA-256 checksums of
every raw file are written by `download_data.py` to
`data/data_coverage_report.md` and `data/raw/manifest.json`.

Three data properties were verified empirically in Phase 1 rather than assumed:

1. **Adjusted prices are genuine total-return series.** Adjusted-close returns
   were reconstructed independently from raw closing prices and the dividend
   record. Yahoo's back-adjustment convention is
   `r_t = Close_t / (Close_{t-1} - Div_t) - 1`, which reproduces the adjusted
   series to within ~2e-6 (float32 storage precision of the adjusted field).
   The naive `(Close_t + Div_t) / Close_{t-1}` convention does *not* match and
   is not used.
2. **Trading calendars are identical across all four instruments.** Over
   2007-01-01 to 2026-09-30 the union and the intersection of trading dates
   both contain 4,967 days, so there are **no genuine missing observations** and
   no instrument-specific gaps to fill. Non-trading days are absent from the
   panel by construction and are never interpolated or forward-filled.
3. **A spurious corporate action exists in the source metadata and must not be
   applied.** Yahoo reports a 10:1 stock split for SLV on 2008-07-24. The SLV
   price series is continuous across that date (close 17.19 → 17.26, +0.4%) and
   neither Yahoo's adjusted close nor `yfinance`'s auto-adjustment applies the
   split. The record is erroneous metadata. The pipeline therefore takes
   split handling from the adjusted-close series only and **never** applies the
   `Stock Splits` column itself. A regression test enforces price continuity
   across this date.

Missing data are distinguished from non-trading days by taking the set of
trading dates from the instruments themselves and testing for date-wise
agreement, rather than by reindexing onto a synthetic calendar. A date present
for some instruments but absent for others would be reported as genuine
missing data; none occurs.

4. **The metal series agree with independent issuers.** SLV and GLD daily
   returns were cross-checked against competing ETFs on the same metal from
   different issuers (SIVR and IAU): return correlations of 0.998 and 0.997 with
   maximum daily gaps of 1.5 and 1.1 percentage points over the overlapping
   history. The peers are used for verification only and never enter the
   analysis.

**Substantive data note.** The sample contains an exceptional precious-metals
boom and crash in its final year: SLV roughly doubled between November 2025 and
February 2026 and fell 28.5% on 2026-01-30 alone, with GLD down 10.3% the same
day. These moves were verified against three independent issuers (SIVR, IAU,
PSLV) and are genuine. The episode falls inside the out-of-sample evaluation
window and will materially affect both the Expected Shortfall estimates that
drive allocation and the realised net returns. It is recorded here before any
results are computed so that it cannot later be treated as a reason to revise
the design. `data/data_coverage_report.md` lists the largest daily moves in
full.

**Non-fabrication rule.** No observation is interpolated, forward-filled,
back-filled, simulated or otherwise invented at any stage. If data required by
this protocol cannot be obtained, the pipeline fails loudly rather than
substituting a proxy.

**Point-in-time caveat.** Yahoo's adjusted prices are back-adjusted and
anchored to the most recent distribution, so the *level* of the adjusted series
is rescaled as new distributions arrive (AGG's adjusted close on 2026-09-30 sits
0.35% below its raw close because of the subsequent October distribution).
Daily *returns within a fixed download* are unaffected, but the series is not
level-stable across downloads. The raw download is therefore archived and
checksummed so that every reported result is reproducible from frozen inputs.

---

## 10. Unresolved decisions

These are genuinely open and must be settled explicitly before the phase that
depends on them. Each is listed with the recommendation that will be adopted
if no other instruction is given. None may be chosen on the basis of the
results it produces.

| # | Decision | Depends on | Recommendation |
|---|---|---|---|
| **U1** | Does "January 2007" mark the start of the *data* sample or of the *evaluation* sample? | Phase 2 | Data sample starts 2007-01; the first 756 trading days are consumed by the estimation window; out-of-sample evaluation runs ≈2010-01 to 2026-09. Rejected alternative: extending the burn-in back to SLV inception (2006-04-28) to start evaluation in 2009, which would change the stated sample period. |
| **U2** | Decision/execution timing at a rebalance. | Phase 2 | Decide on the last trading day of month *M* using returns up to and including that close; execute at the close of the **first trading day of month M+1**. This cannot trade at a price unavailable at the decision time. Executing at the same close is the common convention but is mildly simultaneous. |
| **U3** | Risk-free rate for the Sharpe ratio; Sharpe on excess or raw returns. | Phase 2/3 | Not specified in the approved protocol. Recommend 3-month US Treasury bill (FRED `DTB3`) and Sharpe on excess returns, which also supplies the cash leg of benchmark 4. Requires approval of a second data source. |
| **U4** | Volatility-managed benchmark: target level, volatility estimator and window, cap. | Phase 2 | Prespecify a 10% annualised target, volatility from a 63-trading-day rolling window of the 60/40 portfolio, monthly rebalanced, risky weight capped at 1.0 (no leverage). All values fixed before estimation and not tuned. |
| **U5** | Candidate return construction inside the estimation window. | Phase 2 | Daily returns of fixed target weights (daily rebalanced within the window). Simple and avoids path dependence; the alternative (buy-and-hold weight drift inside the window) makes ES depend on the window's starting point. |
| **U6** | Turnover definition. | Phase 2/3 | One-way turnover: sum of absolute weight changes divided by two, reported annualised. Transaction costs are charged on the full sum of absolute weight changes. |
| **U7** | Block bootstrap: block length, replications, seed. | Phase 3 | Stationary/circular block bootstrap, expected block length 21 trading days (one month), 10,000 replications, seed fixed at 20261025 and recorded in code. |
| **U8** | Archiving raw Yahoo Finance data in a public repository. | Now | Yahoo's terms of service contemplate personal, non-commercial use, so redistribution is not clearly licensed. Reproducibility nonetheless requires frozen inputs. Raw CSVs are currently committed together with checksums; `download_data.py` reproduces them exactly from the source. **If this is unacceptable, delete `data/raw/*.csv` and retain `manifest.json`** — the checksums alone still pin the inputs. |
| **U9** | Do the static benchmarks bear rebalancing costs? | Phase 2 | Yes. All strategies, including static 60/40 and the static metal sleeves, are rebalanced monthly to target and charged the same per-dollar cost. Charging the dynamic strategy alone would bias the primary comparison against it. |
| **U10** | Dividend reinvestment timing. | Accepted | Inherited from the adjusted-price convention: continuous reinvestment at the closing price, with no cash drag between ex-date and payment date. Standard, and uniform across instruments. |

## 11. Phase plan

* **Phase 1 — data acquisition, validation, frozen protocol.** `research_protocol.md`,
  `download_data.py`, `test_data.py`. *This phase.*
* **Phase 2 — portfolio simulation, ES selection, core performance metrics.** `analysis.py`.
  Not to begin before Phase 1 is reviewed and approved.
* **Phase 3 — statistical inference, robustness, publication tables and figures.**

## 12. Reproducibility

* Two scripts and one test file; no classes, services, databases or caching layers.
* All parameters are module-level named constants with documented units.
* Fixed random seeds wherever randomness enters (Phase 3).
* Package versions pinned in `requirements.txt` and recorded in the coverage report.
* Raw inputs checksummed; every transformation applied to them is in
  `download_data.py` and nowhere else.

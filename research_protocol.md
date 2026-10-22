# Research Protocol (Frozen)

**Paper:** Does Dynamic Safe-Haven Allocation Improve Net Portfolio Returns?
**Conference deadline:** 25 October 2026
**Protocol frozen:** 20 October 2026, prior to any performance estimation.
**Revised:** 20 October 2026, following independent review of Phase 1. The
revision corrects a factual error about an SLV corporate action (Section 9) and
settles the implementation decisions that were previously open (Section 10).
No performance results had been computed at the time of either version.
**Revised:** 22 October 2026, following independent review of Phase 3. The
revision records the approved **directional** form of the primary hypothesis in
Section 8, which the Phase 3 code had implemented only as a two-sided test of
equality. This was made after results were known, so it is recorded here
explicitly. It changes no data, no selection rule, no weight, no cost
assumption, no estimation window and no empirical quantity; it changes only
which hypothesis is tested against the same estimated difference. It cannot
make the conclusion more favourable: the estimated difference against the
traditional 60/40 portfolio is **negative**, so a one-sided superiority test can
only fail to reject, and it does (p = 0.5839).
**Status:** Phases 1, 2 and 3 implemented. Manuscript not written.

This document fixes the research design *before* out-of-sample results are
computed. It is the reference against which later code is checked. Changes to
Sections 1-10 after this freeze must be recorded with a reason that is
independent of the results obtained.

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

The risk-free rate and the cash leg of the volatility-managed benchmark use
FRED series `DGS3MO` (3-Month Treasury Constant Maturity Rate). No other
instrument or data series enters the study.

## 3. Sample period

* **Return sample:** 2007-01-03 through 2026-09-30 (daily), 4,967 observations.
* **Price panel:** 2006-12-29 through 2026-09-30. The first date supplies the
  base price for the 2007-01-03 return only and is not itself an observation.
* **Out-of-sample evaluation period:** January 2007 is the start of the **data**
  sample, not of the evaluation sample. The first 756 trading days are consumed
  by the estimation window, so evaluation begins at the first monthly decision
  date at which a full window is available — the first month-end on or after the
  756th trading day of the return sample, which falls in **January 2010** — and
  ends 2026-09-30 (decision D1).

Validation outcome: all four instruments trade continuously over the whole
window. SLV, the shortest series, has an inception date of 2006-04-28, which
precedes the sample start, so **no truncation of the sample period is
required**. The sample period stated in the approved protocol is retained in
full.

## 4. Prices and returns

* Daily **adjusted total-return** prices, taken from the Yahoo Finance
  `Adj Close` field, which is back-adjusted for distributions and splits and
  therefore represents a dividend-reinvested total-return index.
* Daily simple returns: `r_t = P_t / P_{t-1} - 1`, computed only between
  consecutive observed trading days.
* No currency conversion (all instruments are USD-denominated).
* Reinvestment is continuous and at the closing price, the convention embedded
  in the adjusted series (decision D10).

The adjusted-price convention was verified empirically rather than assumed; see
Section 9 and `data/data_coverage_report.md`.

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
* **Decision frequency:** monthly, on the last trading day of each month.
* At each decision date, construct the historical daily return series of each of
  the three candidates over the trailing 756 trading days, compute each
  candidate's **historical 95% Expected Shortfall (ES)**, and select the
  candidate with the **lowest** ES.
* **ES definition (prespecified).** ES is the arithmetic mean of the worst
  `k = floor(0.05 x 756) = 37` daily returns in the window. Historical and
  non-parametric; no distributional assumption.
* **Candidate return series inside the window (decision D5).** Candidate returns
  are constructed on the same basis as the implemented strategy: target weights
  are restored monthly and drift with relative returns between rebalances. The
  estimation input therefore matches the object being traded.
* Only information available at the decision date is used. The selection rule
  never sees returns realised after the decision date, and never uses future
  realised performance of the candidates.

**Decision and execution timing (decision D2).** The decision on the last
trading day of month *M* uses returns up to and including that day's close. The
resulting allocation is executed at the close of the **first trading day of
month M+1**. Between the two dates the portfolio continues to hold the
previously selected allocation and earns its returns, and the trade executed at
the later close is the difference between the drifted held weights and the newly
selected target weights. Transaction costs are charged on that actual traded
change, at execution. Nothing is ever traded at a price unavailable when the
decision was made.

## 7. Transaction costs

* **Principal assumption:** 10 basis points per traded dollar.
* **Robustness:** 25 and 50 basis points.
* Costs are charged on the sum of absolute changes in position weights at each
  execution, and are applied to the dynamic strategy **and** to every benchmark
  on the same per-dollar basis (decision D9). Charging only the dynamic strategy
  would bias the primary comparison against it.
* **Turnover (decision D6):** reported as one-way turnover, the sum of absolute
  weight changes divided by two, annualised. Costs are charged on the full sum
  of absolute weight changes, not the halved figure.

## 8. Benchmarks and outcomes

**Benchmarks**

1. Traditional 60/40 (SPY/AGG), monthly rebalanced to target.
2. Static gold: 57/38/5 SPY/AGG/GLD, monthly rebalanced to target.
3. Static silver: 57/38/5 SPY/AGG/SLV, monthly rebalanced to target.
4. Conventional volatility-managed 60/40 (decision D4): **10% annualised
   volatility target**; realised volatility estimated from the **preceding 63
   trading days** of 60/40 daily returns, using only data strictly before the
   decision date, annualised by `sqrt(252)`; risky weight
   `w = min(1.0, 0.10 / sigma_hat)`, so exposure is **capped at 100% and never
   levered**; the residual `1 - w` is held in cash at the risk-free rate.
   Rebalanced monthly on the same decision and execution dates as the dynamic
   strategy.

All benchmarks follow the same decision/execution timing and cost treatment as
the dynamic strategy.

**Outcomes**

* *Primary:* out-of-sample **net portfolio return**.
* *Secondary:* Sharpe ratio, 95% Expected Shortfall, maximum drawdown, portfolio
  turnover.

Definitions, prespecified: the Sharpe ratio is computed on daily **excess**
returns over the risk-free rate and annualised as
`252 x mean(excess) / (sqrt(252) x sd(excess))`; realised ES uses the same
5%-tail mean as Section 6, applied to the realised out-of-sample net return
series; maximum drawdown is the largest peak-to-trough decline of the cumulative
net return series.

**Risk-free rate (decision D3).** FRED `DGS3MO`, published as an annualised
percentage on an investment basis, converted to a daily rate as
`(1 + y/100) ** (1/252) - 1`. The rate dated *t-1* is used for the cash return
earned on day *t*, so no yield enters a decision or a return before it was
published. On the ~2 trading days per year when the bond market is closed but
the NYSE trades (Columbus Day, Veterans Day) the last previously published yield
is carried forward; every occurrence is flagged in
`data/processed/risk_free_daily.csv` and counted in the coverage report.

**Inference (decision D7)**

Paired out-of-sample net return differences (dynamic strategy minus each
benchmark), evaluated with a **stationary block bootstrap**, expected block
length **21 trading days** (one month), **10,000 replications**, random seed
fixed at **20261025** and set explicitly in code. All comparisons are drawn on
identical replications.

*Primary hypothesis (directional).* The approved primary hypothesis is a test
of **superiority** of the dynamic strategy over the traditional 60/40 portfolio,
on the primary outcome (net return) at the principal cost assumption of 10 basis
points:

> H0: E[r_dynamic,t - r_60/40,t] <= 0
> H1: E[r_dynamic,t - r_60/40,t] >  0

It is one-sided and is reported with a one-sided 95% **lower confidence bound**
(the upper limit is unbounded). This is a single prespecified hypothesis and
carries no multiplicity burden.

*Supplementary comparisons.* Every other comparison — the remaining benchmarks,
and all comparisons at the 25 and 50 basis point robustness levels — is a
**two-sided** test of equality (H0: difference = 0) reported with a two-sided
percentile interval. These form a family of three testable comparisons per cost
level; a Bonferroni-adjusted p-value is reported alongside them as context only,
and was not prespecified.

Where two strategies produce *exactly* identical return series, the difference
is identically zero, the bootstrap is degenerate and no p-value is reported,
under either alternative.

---

## 9. Data provenance and verification

**Sources.** Yahoo Finance, accessed through the `yfinance` package, for the
four instruments and the two verification peers. FRED, as CSV, for `DGS3MO`.
Download date, package versions, row counts, coverage and SHA-256 checksums of
every raw file are written by `download_data.py` to
`data/data_coverage_report.md` and `data/raw/manifest.json`.

Four data properties were verified empirically in Phase 1 rather than assumed:

1. **Adjusted prices are genuine total-return series.** Adjusted-close returns
   were reconstructed independently from raw closing prices and the dividend
   record. Yahoo's back-adjustment convention is
   `r_t = Close_t / (Close_{t-1} - Div_t) - 1`, which reproduces the adjusted
   series to within ~2e-6 (float32 storage precision of the adjusted field). The
   naive `(Close_t + Div_t) / Close_{t-1}` convention does *not* match and is
   not used.

2. **Share splits are adjusted exactly once.** SLV underwent a **genuine
   10-for-1 share split on 2008-07-24**, which reduced the silver backing from
   approximately 10 ounces to approximately 1 ounce per share. The split is
   documented in the trust's SEC filings (Form 10-Q, iShares Silver Trust,
   CIK 0001330568:
   `https://www.sec.gov/Archives/edgar/data/1330568/000119312509106967/d10q.htm`).

   Both of Yahoo's price columns, `Close` and `Adj Close`, are back-adjusted for
   it, which is why the series is continuous across the date. The pipeline
   therefore takes split handling from those price columns and **never applies
   the `Stock Splits` column itself**, so the adjustment is applied exactly
   once. Continuity across the split date is the test that the adjustment is
   present once and only once: an unadjusted series would fall about 90% on that
   date and a doubly-adjusted one would rise about tenfold, and both would
   breach the plausibility bound in `download_data.py`.

   This corrects an error in the first version of this protocol, which read the
   observed continuity as evidence that the corporate action had not occurred.
   Continuity is evidence of adjustment, not of absence. The returns computed
   from `Adj Close` were correct under either reading and are unchanged; only the
   documentation and the name of the corresponding test were wrong. An
   independent check confirms the mechanism: dividing Yahoo's `Close` by the
   silver spot price implies approximately **1.0 ounces per share on both sides
   of the split date** (0.99 in July 2008 before it, 0.99 after), which can only
   hold if the pre-split prices are already divided by ten.

3. **The trading calendar matches an authoritative source.** Agreement among the
   four instruments shows only that they report the same dates; it cannot reveal
   a day missing from all four. The panel is therefore also compared with the
   official **NYSE** trading calendar from `pandas_market_calendars`. Over
   2007-01-03 to 2026-09-30 the panel contains 4,967 days and the NYSE calendar
   contains 4,967 days, with **no date in either set absent from the other**. The
   calendar is used for validation only; the panel's dates always come from the
   instruments themselves and are never constructed from it.

   Exceptional closures (weekdays that were not scheduled holidays) are
   enumerated in `download_data.py` and individually checked to be absent from
   both the calendar and the panel: 2007-01-02 (national day of mourning for
   President Ford), 2012-10-29 and 2012-10-30 (Hurricane Sandy), 2018-12-05
   (national day of mourning for President George H. W. Bush) and 2025-01-09
   (national day of mourning for President Carter).

4. **The metal series agree with independent issuers.** SLV and GLD daily
   returns were cross-checked against competing ETFs on the same metal from
   different issuers (SIVR and IAU): return correlations of 0.998 and 0.997 with
   maximum daily gaps of 1.5 and 1.1 percentage points over the overlapping
   history. The peers are used for verification only and never enter the
   analysis.

**Missing data versus non-trading days.** Genuine missing data would appear
either as a date present for some instruments and absent for others (checked:
none) or as a date in the official exchange calendar that is absent from the
panel (checked: none). Non-trading days are absent from the panel by
construction. No price observation is interpolated, forward filled, back filled,
simulated or otherwise invented at any stage. The single filling operation
anywhere in the pipeline is the carry-forward of the published Treasury yield on
bond-market holidays described in Section 8, which is a convention on the level
of an interest rate rather than an invented price observation, uses only past
information, and is flagged observation by observation.

**Substantive data note.** The sample contains an exceptional precious-metals
boom and crash in its final year: SLV roughly doubled between November 2025 and
February 2026 and fell 28.5% on 2026-01-30 alone, with GLD down 10.3% the same
day. These moves were verified against three independent issuers (SIVR, IAU,
PSLV) and are genuine. The episode falls inside the out-of-sample evaluation
window and will materially affect both the Expected Shortfall estimates that
drive allocation and the realised net returns. It is recorded here before any
results are computed so that it cannot later be treated as a reason to revise
the design. `data/data_coverage_report.md` lists the largest daily moves in full.

**Point-in-time caveat.** Yahoo's adjusted prices are back-adjusted and anchored
to the most recent distribution, so the *level* of the adjusted series is
rescaled as new distributions arrive (AGG's adjusted close on 2026-09-30 sits
0.35% below its raw close because of the subsequent October distribution). Daily
*returns within a fixed download* are unaffected, but the series is not
level-stable across downloads. The raw download is therefore archived and
checksummed, and the default build verifies those checksums instead of
re-downloading, so every reported result is reproducible from frozen inputs.

---

## 10. Implementation decisions

All methodological decisions are now frozen. Each was settled on
design grounds before any performance result was computed, and none may be
revisited on the basis of the results it produces.

| # | Decision | Status |
|---|---|---|
| **D1** | January 2007 begins the **data** sample. The first 756 trading days are consumed by the estimation window; out-of-sample evaluation runs from the first month-end with a full window (**January 2010**) to 2026-09. | Frozen |
| **D2** | Decide on the last trading day of month *M* using data up to and including that close; execute at the close of the **first trading day of month M+1**. The previously selected allocation is held and earns returns between the two dates, and the trade is the difference between those drifted weights and the new targets. | Frozen |
| **D3** | Risk-free rate and cash return: FRED `DGS3MO`, converted to a daily rate by `(1 + y/100) ** (1/252) - 1`, using the rate dated *t-1* for day *t*. Sharpe computed on excess returns. | Frozen |
| **D4** | Volatility-managed benchmark: 10% annualised target, realised volatility from the preceding 63 trading days (strictly prior data only), risky weight `min(1.0, 0.10 / sigma_hat)`, no leverage, residual in cash, monthly rebalancing. | Frozen |
| **D5** | Candidate return series inside the estimation window are built with monthly rebalancing to target weights and drift between rebalances, matching the implemented strategy. | Frozen |
| **D6** | One-way turnover (sum of absolute weight changes divided by two), annualised. Costs charged on the full sum of absolute weight changes. | Frozen |
| **D7** | Stationary block bootstrap, expected block length 21 trading days, 10,000 replications, seed 20261025. | Frozen |
| **D8** | Storage of Yahoo-derived data in this repository. | **Open — repository policy, see Section 11** |
| **D9** | Every strategy, including the static benchmarks, is charged the same cost per dollar traded. | Frozen |
| **D10** | Dividend reinvestment is continuous and at the closing price, inherited from the adjusted-price convention and uniform across instruments. | Frozen |

## 11. Data licensing (D8, open)

Yahoo Finance's terms of service restrict redistribution of its data. This
repository is public and `data/raw/` currently contains Yahoo-derived CSV files,
committed so that results are reproducible from frozen inputs. FRED data are US
Government works and are not subject to copyright, so only the Yahoo-derived
files are at issue.

**Deleting the CSV files in a later commit would not resolve this.** Git retains
every committed version, so the files would remain in the repository's published
history and reachable from earlier commits. Removing published data requires
rewriting history and force-pushing, which changes commit hashes for every
collaborator, or making the repository private.

The options, for the repository owner to decide. No action has been taken on any
of them, and repository visibility has not been changed and history has not been
rewritten:

1. **Make the repository private.** Simplest and immediate. Keeps full
   reproducibility, and the history becomes non-public. A public code-only
   mirror can be published separately if a public artifact is wanted.
2. **Public code-only repository, data distributed separately.** Keep this
   repository's code and `manifest.json` public, and remove the raw CSVs. Because
   deletion does not clear history, this requires history rewriting to be
   effective. Reviewers reproduce by running `--download`, and the manifest
   checksums tell them whether they obtained identical inputs.
3. **Keep the data public as-is.** Accept the licensing exposure for four
   factual ETF price series used in non-commercial academic work. The least
   effort and the highest residual risk.
4. **Keep the data, but in a private store.** Raw files in a private bucket or
   private submodule, referenced from this repository by checksum.

Options 1, 2 and 4 all preserve reproducibility, because `manifest.json` pins
the exact inputs by SHA-256 whether or not the files themselves are public.
Option 2 is the only one requiring history rewriting, and would need explicit
approval before anything is force-pushed.

## 12. Phase plan

* **Phase 1 — data acquisition, validation, frozen protocol.**
  `research_protocol.md`, `download_data.py`, `test_data.py`. *This phase.*
* **Phase 2 — portfolio simulation, ES selection, core performance metrics.**
  `analysis.py`. Not to begin before Phase 1 is reviewed and approved.
* **Phase 3 — statistical inference, robustness, publication tables and figures.**

## 13. Reproducibility

* Two scripts and one test file; no classes, services, databases or caching
  layers.
* All parameters are module-level named constants with documented units.
* **Frozen inputs are never silently overwritten.** `python3 download_data.py`
  rebuilds the processed dataset from the raw files already in `data/raw/`,
  verifying every SHA-256 against `data/raw/manifest.json` and failing without
  writing anything if an input is missing or altered. It makes no network calls
  and is the documented reproduction procedure. Refreshing the raw inputs
  requires `python3 download_data.py --download`, which is explicit about
  replacing the research inputs.
* Fixed random seed 20261025 wherever randomness enters (Phase 3).
* Package versions pinned in `requirements.txt` and recorded in the coverage
  report and manifest.
* Raw inputs checksummed; every transformation applied to them is in
  `download_data.py` and nowhere else, and all of them are listed in the
  coverage report.

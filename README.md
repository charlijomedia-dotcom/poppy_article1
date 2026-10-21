# Does Dynamic Safe-Haven Allocation Improve Net Portfolio Returns?

Replication code for a short conference paper (8-10 pages including
references; deadline 25 October 2026).

**Research question.** Does dynamically allocating between gold, silver and no
metal improve the out-of-sample net return of a conventional 60/40 US
stock-bond portfolio?

The design is frozen in [`research_protocol.md`](research_protocol.md) and was
fixed before any performance estimation. Read that first.

## Status

| Phase | Content | State |
|---|---|---|
| 1 | Data acquisition, validation, frozen protocol | **Complete, awaiting review** |
| 2 | Portfolio simulation, Expected Shortfall selection, performance metrics | Not started |
| 3 | Statistical inference, robustness, tables and figures | Not started |

Phase 2 does not begin until Phase 1 has been independently reviewed and approved.
No portfolio-performance results have been computed.

## Layout

```
research_protocol.md               frozen research design and implementation decisions
download_data.py                   Phase 1: build, validate and freeze the dataset
test_data.py                       structure, chronology and integrity tests
requirements.txt                   pinned dependencies
data/raw/                          frozen raw downloads + SHA-256 manifest
data/processed/                    price panel, daily returns, risk-free series
data/data_coverage_report.md       generated coverage and validation report
```

## Reproducing Phase 1

```bash
pip install -r requirements.txt
python3 download_data.py            # rebuild from frozen inputs; no network calls
python3 -m pytest test_data.py -v
```

The default build **rebuilds the processed dataset from the frozen raw files in
`data/raw/`**, verifying every SHA-256 against `data/raw/manifest.json`. If an
input is missing or altered it fails without writing anything, so the
reproduction path can never silently overwrite the research inputs.

To deliberately refresh the raw data from the sources:

```bash
python3 download_data.py --download  # replaces data/raw/ and the manifest
```

Both modes validate rather than repair: no price observation is interpolated,
forward filled or simulated anywhere in the pipeline. The script exits non-zero
if any check fails. `test_data.py` reads the frozen artifacts, makes no network
calls, and fails (rather than skipping) if a required file is absent.

## Data

Daily adjusted total-return prices for SPY, AGG, GLD and SLV from Yahoo
Finance, covering 2007-01-03 to 2026-09-30 — 4,967 daily return observations per
instrument, matching the official NYSE trading calendar exactly, with no missing
observations. The risk-free rate and cash leg use FRED series `DGS3MO`. SIVR and
IAU are downloaded as independent cross-checks on the metal series and never
enter the analysis.

Coverage, provenance, download timestamps, the full list of transformations and
SHA-256 checksums are in
[`data/data_coverage_report.md`](data/data_coverage_report.md).

## Data licensing (open question)

Yahoo Finance's terms of service restrict redistribution, this repository is
public, and `data/raw/` contains Yahoo-derived CSV files. Deleting them in a
later commit would **not** remove them from the published Git history. The
options — private repository, public code-only repository with history rewritten,
private data store, or accepting the exposure — are set out in
[`research_protocol.md`](research_protocol.md) §11 as decision **D8**, for the
repository owner to decide. Repository visibility has not been changed and Git
history has not been rewritten.

FRED data are US Government works and are not subject to copyright.

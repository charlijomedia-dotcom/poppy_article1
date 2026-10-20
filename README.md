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

## Layout

```
research_protocol.md            frozen research design and unresolved decisions
download_data.py                Phase 1: download, validate and freeze the data
test_data.py                    structure, chronology and integrity tests
requirements.txt                pinned dependencies
data/raw/                       unmodified Yahoo Finance download + checksums
data/processed/                 aligned adjusted price panel and daily returns
data/data_coverage_report.md    generated coverage and validation report
```

## Reproducing Phase 1

```bash
pip install -r requirements.txt
python3 download_data.py        # downloads, validates, writes the report
python3 -m pytest test_data.py -v
```

`download_data.py` exits non-zero if any validation check fails. It validates
rather than repairs: no observation is interpolated, forward filled or
simulated anywhere in the pipeline. `test_data.py` reads the frozen artifacts
and makes no network calls.

## Data

Daily adjusted total-return prices for SPY, AGG, GLD and SLV from Yahoo
Finance, covering 2007-01-03 to 2026-09-30 (4,967 daily return observations per
instrument, with no missing observations). Coverage, provenance, download
timestamps, transformations and SHA-256 checksums are in
[`data/data_coverage_report.md`](data/data_coverage_report.md).

Yahoo Finance's terms of service contemplate personal, non-commercial use, so
archiving the raw CSVs here is a judgement call recorded as unresolved decision
**U8** in the protocol. The checksums alone pin the inputs if the CSVs must be
removed.

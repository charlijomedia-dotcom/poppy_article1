"""Phase 3: statistical inference, cost robustness, publication tables and figures.

Run:  python3 inference.py

Reads the frozen Phase 1 dataset and re-runs the approved Phase 2 simulation
(no downloads, nothing re-estimated). Writes tables, figures and inference
results to data/results/.

Implements research_protocol.md decision D7 as frozen: a stationary block
bootstrap with expected block length 21 trading days, 10,000 replications and
seed 20261025, applied to paired daily net-return differences. The portfolio
methodology is imported from analysis.py and is not re-specified here.

The dynamic strategy selected gold at every monthly decision, so its net return
series is *exactly* identical to static gold. That comparison has no sampling
variation and no testable hypothesis, so no p-value is reported for it; see
`paired_comparison`.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as an

# --- Prespecified inference parameters (research_protocol.md D7) -------------

BLOCK_LENGTH = 21            # expected block length, trading days
N_BOOTSTRAP = 10_000         # replications
SEED = 20261025              # fixed random seed
CONFIDENCE = 0.95

# Transaction-cost levels: the principal assumption plus the two prespecified
# robustness levels (research_protocol.md section 7).
PRINCIPAL_COST_BPS = 10
COST_LEVELS_BPS = [10, 25, 50]

# Risk-free series identifier (research_protocol.md decision D3). Named here
# rather than imported from download_data.py, which would pull in the download
# dependencies that Phase 3 deliberately does not need.
RISK_FREE_SERIES_ID = "DGS3MO"

BENCHMARKS = ["traditional_60_40", "static_gold", "static_silver",
              "volatility_managed_60_40"]
DYNAMIC = "dynamic_es"

STRATEGY_LABELS = {
    "traditional_60_40": "Traditional 60/40",
    "static_gold": "Static gold",
    "static_silver": "Static silver",
    "dynamic_es": "Dynamic ES",
    "volatility_managed_60_40": "Vol-managed 60/40",
}
CANDIDATE_LABELS = {"no_metal": "No metal", "gold": "Gold", "silver": "Silver"}

RESULTS_DIR = an.RESULTS_DIR

# Categorical palette, fixed order, never cycled. Validated for colour-vision
# deficiency separation, chroma, lightness band and contrast against a light
# surface. Every series also carries a distinct dash pattern and a direct label,
# so identity survives greyscale printing and is never carried by colour alone.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#7E4E9B"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"


# --- Stationary block bootstrap ---------------------------------------------

def stationary_bootstrap_means(series_map, n_boot=N_BOOTSTRAP,
                               expected_block=BLOCK_LENGTH, seed=SEED,
                               chunk=2500):
    """Bootstrap distributions of the sample mean, by the stationary bootstrap.

    Politis and Romano (1994): blocks start at a uniformly random observation
    and each subsequent step either continues the block or starts a new one with
    probability 1 / expected_block, wrapping circularly. Block lengths are
    therefore geometric with mean `expected_block`, which keeps the resampled
    series stationary.

    All series in `series_map` are resampled with the *same* draws, so
    comparisons across benchmarks and across cost levels use identical
    replications. Returns {name: array of n_boot bootstrap means}.
    """
    arrays = {name: np.asarray(values, dtype=float)
              for name, values in series_map.items()}
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"series must share one length, got {lengths}")
    n = lengths.pop()
    if n < expected_block:
        raise ValueError(f"series of length {n} is shorter than the block length")

    restart_probability = 1.0 / expected_block
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n_boot) for name in arrays}

    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        index = bootstrap_index_block(n, size, restart_probability, rng)
        for name, values in arrays.items():
            out[name][done:done + size] = values[index].mean(axis=1)
        done += size
    return out


def bootstrap_index_block(n, size, restart_probability, rng):
    """One block of stationary-bootstrap resampling indices, shape (size, n).

    Each row starts at a uniformly random observation. At every later step the
    row either continues the current block, advancing by one position, or starts
    a new block at a fresh uniform draw with probability `restart_probability`.
    Positions wrap circularly, so every observation has equal selection
    probability and block lengths are geometric with mean
    1 / restart_probability.
    """
    index = np.empty((size, n), dtype=np.int64)
    index[:, 0] = rng.integers(0, n, size)
    restart = rng.random((size, n)) < restart_probability
    fresh = rng.integers(0, n, (size, n))
    for t in range(1, n):
        index[:, t] = np.where(restart[:, t], fresh[:, t], index[:, t - 1] + 1)
    np.mod(index, n, out=index)
    return index


def paired_comparison(dynamic_net, benchmark_net, boot_means=None,
                      confidence=CONFIDENCE):
    """Test the paired daily net-return difference against zero.

    Hypotheses, two-sided:
        H0: E[d_t] = 0      the dynamic strategy and the benchmark have equal
                            expected daily net returns
        H1: E[d_t] != 0
    where d_t = net return of the dynamic strategy minus that of the benchmark
    on the same day. Pairing by date removes the common market component, so the
    test speaks only to the difference the allocation rule makes.

    Inference is by the stationary block bootstrap: the confidence interval is
    the percentile interval of the bootstrap means, and the p-value is the share
    of replications whose mean, recentred on the null, is at least as extreme as
    the observed mean. The (1 + count) / (1 + B) convention keeps the p-value
    strictly positive and makes its resolution explicit.

    If the two series are *exactly* identical the difference is identically
    zero: there is no sampling variation, the bootstrap is degenerate, and a
    numerical p-value would be meaningless. Such a comparison is returned with
    `status = "identical_series"` and no p-value.
    """
    a = np.asarray(dynamic_net, dtype=float)
    b = np.asarray(benchmark_net, dtype=float)
    diff = a - b
    n = len(diff)
    scale = an.TRADING_DAYS_PER_YEAR

    if np.array_equal(a, b):
        return {
            "status": "identical_series",
            "n_days": n,
            "mean_daily_difference": 0.0,
            "annualized_difference": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": np.nan,
            "p_value_resolution": np.nan,
            "information_ratio": np.nan,
            "n_days_differing": 0,
            "max_abs_daily_difference": 0.0,
        }

    observed = float(diff.mean())
    if boot_means is None:
        boot_means = stationary_bootstrap_means({"d": diff})["d"]
    boot_means = np.asarray(boot_means, dtype=float)

    alpha = 1.0 - confidence
    low, high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Recentre the bootstrap distribution on the null before counting.
    extreme = np.abs(boot_means - observed) >= abs(observed)
    p_value = (1 + int(extreme.sum())) / (1 + len(boot_means))

    sd = float(diff.std(ddof=1))
    return {
        "status": "tested",
        "n_days": n,
        "mean_daily_difference": observed,
        "annualized_difference": observed * scale,
        "ci_low": float(low) * scale,
        "ci_high": float(high) * scale,
        "p_value": p_value,
        "p_value_resolution": 1.0 / (1 + len(boot_means)),
        "information_ratio": (
            float(np.sqrt(scale) * observed / sd) if sd > 0 else np.nan
        ),
        "n_days_differing": int((diff != 0).sum()),
        "max_abs_daily_difference": float(np.abs(diff).max()),
    }


# --- Simulation across cost levels ------------------------------------------

def run_cost_level(returns, rf_prev, schedule, cost_bps):
    """Run all five strategies at one transaction-cost level.

    `build_strategies` is rebuilt per cost level so its allocation record is not
    shared between runs. The Expected Shortfall selection uses gross candidate
    returns and so does not depend on the cost level; this is verified in
    `main`, not assumed.
    """
    strategies, allocations = an.build_strategies(returns, schedule)
    histories = {
        name: an.simulate(returns, rf_prev, schedule, target,
                          cost_per_dollar=cost_bps / 10_000.0)
        for name, target in strategies.items()
    }
    metrics = pd.DataFrame(
        {name: an.performance_metrics(h, rf_prev) for name, h in histories.items()}
    ).T
    metrics.index.name = "strategy"
    daily = pd.DataFrame({name: h["net_return"] for name, h in histories.items()})
    daily.index.name = "Date"
    return histories, metrics, daily, allocations


# --- Task 4: selection behaviour --------------------------------------------

def selection_summary(allocations):
    """How often each candidate is selected, and the Expected Shortfall margin.

    Expected Shortfall is negative (a loss), and the rule selects the candidate
    whose shortfall is least severe, i.e. the largest (least negative) value.
    The margin is the selected candidate's Expected Shortfall minus the best
    rejected one, so a positive margin means the selection was strict.
    """
    frame = pd.DataFrame(allocations)
    es_cols = [f"es_{name}" for name in an.CANDIDATES]

    counts = (
        frame["selected"].value_counts().reindex(list(an.CANDIDATES)).fillna(0).astype(int)
    )
    frequency = pd.DataFrame({
        "candidate": [CANDIDATE_LABELS[c] for c in counts.index],
        "decisions_selected": counts.to_numpy(),
        "share_of_decisions": counts.to_numpy() / len(frame),
    })

    selected_es = frame.apply(lambda r: r[f"es_{r['selected']}"], axis=1)
    rejected_best = frame.apply(
        lambda r: max(r[f"es_{c}"] for c in an.CANDIDATES if c != r["selected"]),
        axis=1,
    )
    margin = selected_es - rejected_best
    runner_up = frame.apply(
        lambda r: max(
            (c for c in an.CANDIDATES if c != r["selected"]),
            key=lambda c: r[f"es_{c}"],
        ),
        axis=1,
    )

    advantage = pd.DataFrame({
        "decision_date": frame["decision_date"],
        "selected": frame["selected"],
        "runner_up": runner_up,
        "es_selected": selected_es,
        "es_runner_up": rejected_best,
        "es_margin": margin,
        "es_margin_bp_per_day": margin * 10_000,
        **{col: frame[col] for col in es_cols},
    })

    quantiles = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]
    summary = pd.DataFrame({
        "statistic": (["mean", "sd"] + [f"q{int(q * 100):02d}" for q in quantiles]
                      + ["share_strictly_positive"]),
        "es_margin_bp_per_day": (
            [margin.mean() * 1e4, margin.std(ddof=1) * 1e4]
            + [margin.quantile(q) * 1e4 for q in quantiles]
            + [float((margin > 0).mean())]
        ),
    })
    return frequency, advantage, summary


# --- Tables ------------------------------------------------------------------

def descriptive_table(returns, risk_free_raw, evaluation_index):
    """Table 1: instrument-level descriptive statistics over the data sample."""
    rows = []
    for ticker in an.ASSETS:
        series = returns[ticker]
        rows.append({
            "Instrument": ticker,
            "Obs.": len(series),
            "Start": str(series.index.min().date()),
            "End": str(series.index.max().date()),
            "Ann. mean (%)": series.mean() * an.TRADING_DAYS_PER_YEAR * 100,
            "Ann. s.d. (%)": series.std(ddof=1) * np.sqrt(an.TRADING_DAYS_PER_YEAR) * 100,
            "Skew": series.skew(),
            "Exc. kurtosis": series.kurtosis(),
            "Min (%)": series.min() * 100,
            "Max (%)": series.max() * 100,
            "ES 95% (%)": an.expected_shortfall(series) * 100,
        })
    panel_a = pd.DataFrame(rows).set_index("Instrument")

    panel_b = returns.corr()
    panel_b.index.name = "Instrument"

    rf = risk_free_raw.loc[returns.index]
    notes = pd.DataFrame({
        "item": [
            "Data sample (daily returns)",
            "Observations per instrument",
            "Out-of-sample evaluation window",
            "Evaluation observations",
            "Risk-free series",
            "Risk-free annualised yield range (%)",
        ],
        "value": [
            f"{returns.index.min().date()} to {returns.index.max().date()}",
            str(len(returns)),
            f"{evaluation_index.min().date()} to {evaluation_index.max().date()}",
            str(len(evaluation_index)),
            f"FRED {RISK_FREE_SERIES_ID} (3-month Treasury constant maturity)",
            f"{rf.min():.2f} to {rf.max():.2f}",
        ],
    })
    return panel_a, panel_b, notes


def performance_table(metrics):
    """Table 2: out-of-sample performance at the principal cost assumption."""
    frame = pd.DataFrame({
        "Strategy": [STRATEGY_LABELS[s] for s in metrics.index],
        "Ann. mean net return (%)": metrics["annualized_net_return"] * 100,
        "CAGR (%)": metrics["cagr"] * 100,
        "Ann. s.d. (%)": metrics["annualized_volatility"] * 100,
        "Sharpe": metrics["sharpe_ratio"],
        "ES 95% (%)": metrics["expected_shortfall_95"] * 100,
        "Max drawdown (%)": metrics["max_drawdown"] * 100,
        "Turnover (1-way, p.a.)": metrics["annualized_turnover_one_way"],
        "Cost drag (bp p.a.)": metrics["annualized_cost_drag"] * 10_000,
    }).set_index("Strategy")
    return frame


def inference_table(inference, performance_by_cost):
    """Table 3: paired comparisons (Panel A) and cost robustness (Panel B)."""
    rows = []
    for record in inference:
        if record["status"] == "identical_series":
            ci = "--"
            p = "n/a"
            effect = "0.000"
            ir = "--"
        else:
            ci = f"[{record['ci_low'] * 100:.3f}, {record['ci_high'] * 100:.3f}]"
            p = (f"{record['p_value']:.4f}" if record["p_value"] > record["p_value_resolution"]
                 else f"<{record['p_value_resolution']:.4f}")
            effect = f"{record['annualized_difference'] * 100:.3f}"
            ir = f"{record['information_ratio']:.3f}"
        rows.append({
            "Cost (bp)": record["cost_bps"],
            "Comparison": f"Dynamic ES - {STRATEGY_LABELS[record['benchmark']]}",
            "Ann. difference (pp)": effect,
            "95% bootstrap CI (pp)": ci,
            "Info. ratio": ir,
            "p-value": p,
            "Status": record["status"],
        })
    panel_a = pd.DataFrame(rows)

    panel_b = pd.DataFrame({
        "Strategy": [STRATEGY_LABELS[s] for s in performance_by_cost[PRINCIPAL_COST_BPS].index],
    }).set_index("Strategy")
    for cost in COST_LEVELS_BPS:
        metrics = performance_by_cost[cost]
        panel_b[f"{cost} bp: ann. net (%)"] = (
            metrics["annualized_net_return"] * 100
        ).to_numpy()
        panel_b[f"{cost} bp: Sharpe"] = metrics["sharpe_ratio"].to_numpy()
    return panel_a, panel_b


def write_latex(frame, path, caption, label, float_format="{:.3f}", index=True,
                note=None):
    """Write a booktabs LaTeX table. Editable plain text, no packages beyond booktabs."""
    def fmt(value):
        if isinstance(value, float) and np.isfinite(value):
            return float_format.format(value)
        text = str(value)
        return text.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")

    body = frame.reset_index() if index else frame
    headers = [str(c).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
               for c in body.columns]
    align = "l" + "r" * (len(headers) - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in body.iterrows():
        lines.append(" & ".join(fmt(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\begin{{minipage}}{{\linewidth}}\footnotesize {note}\end{{minipage}}")
    lines += [r"\end{table}", ""]
    Path(path).write_text("\n".join(lines))


# --- Figures -----------------------------------------------------------------

def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)


def figure_cumulative_wealth(daily, path_stem):
    """Figure 1: cumulative net wealth, starting from an initial wealth of 1."""
    order = ["traditional_60_40", "static_silver", "volatility_managed_60_40",
             "static_gold"]
    dashes = {"traditional_60_40": (None, None), "static_silver": (5, 1.6),
              "volatility_managed_60_40": (1.4, 1.4),
              "static_gold": (7, 2, 1.4, 2)}

    wealth = {}
    start = daily.index.min() - pd.Timedelta(days=1)
    for name in daily.columns:
        path = (1.0 + daily[name]).cumprod()
        wealth[name] = pd.concat([pd.Series([1.0], index=[start]), path])

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    _style_axes(ax)

    for colour, name in zip(PALETTE, order):
        terminal = wealth[name].iloc[-1]
        ax.plot(wealth[name].index, wealth[name].to_numpy(), color=colour,
                linewidth=1.5, dashes=dashes[name],
                label=f"{STRATEGY_LABELS[name]}  ({terminal:.2f}x)",
                solid_capstyle="round")

    # The dynamic rule selected gold at every decision, so its wealth path lies
    # exactly on static gold's. Drawn as a thin overlay so the coincidence is
    # visible rather than hidden underneath.
    ax.plot(wealth[DYNAMIC].index, wealth[DYNAMIC].to_numpy(), color=INK,
            linewidth=0.7, dashes=(1, 2.4),
            label=f"Dynamic ES  ({wealth[DYNAMIC].iloc[-1]:.2f}x, identical "
                  f"to static gold)")

    ax.set_ylabel("Cumulative net wealth (initial wealth = 1)", fontsize=9.5,
                  color=INK)
    ax.set_xlabel("")
    ax.set_title("Out-of-sample cumulative net wealth, 10 bp transaction cost",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.axhline(1.0, color=MUTED, linewidth=0.7, alpha=0.7)

    # No right-edge direct labels here: the four fully invested strategies end
    # within two percent of one another, so the labels would overlap. Identity
    # rests on the legend plus a distinct dash pattern per series, which also
    # survives greyscale printing.
    ax.set_xlim(wealth[order[0]].index.min(),
                wealth[order[0]].index.max() + pd.Timedelta(days=40))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              labelcolor=INK, handlelength=3.2)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(f"{path_stem}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_es_differences(advantage, path_stem):
    """Figure 2: Expected Shortfall advantage of gold at each decision date.

    A plot of allocation switching would be uninformative, because the rule
    never switched. What varies, and what drove every decision, is the margin
    by which gold's Expected Shortfall beat each alternative.
    """
    dates = pd.to_datetime(advantage["decision_date"])
    versus_no_metal = (advantage["es_gold"] - advantage["es_no_metal"]) * 10_000
    versus_silver = (advantage["es_gold"] - advantage["es_silver"]) * 10_000

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    _style_axes(ax)

    ax.axhline(0.0, color=INK, linewidth=0.9)
    ax.plot(dates, versus_no_metal.to_numpy(), color=PALETTE[0], linewidth=1.5,
            label="Gold vs. no metal")
    ax.plot(dates, versus_silver.to_numpy(), color=PALETTE[1], linewidth=1.5,
            dashes=(5, 1.6), label="Gold vs. silver")

    ax.set_ylabel("Expected Shortfall advantage of gold\n(basis points per day)",
                  fontsize=9.5, color=INK)
    ax.set_title(
        "Why the rule never switched: gold's 95% Expected Shortfall advantage",
        fontsize=10.5, color=INK, loc="left", pad=10,
    )
    ax.annotate(
        "above zero: gold's shortfall is the least severe, so gold is selected",
        xy=(0.015, 0.03), xycoords="axes fraction", fontsize=8, color=MUTED,
        va="bottom",
    )
    for series, colour, label in [(versus_no_metal, PALETTE[0], "vs. no metal"),
                                  (versus_silver, PALETTE[1], "vs. silver")]:
        ax.annotate(f" {label}", xy=(dates.iloc[-1], series.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=8,
                    color=colour, va="center")
    ax.set_xlim(dates.min(), dates.max() + pd.Timedelta(days=560))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK,
              handlelength=3.2)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(f"{path_stem}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Main --------------------------------------------------------------------

def main():
    returns, rf_prev = an.load_inputs()
    risk_free_raw = pd.read_csv(an.RISK_FREE_PATH, index_col="Date",
                                parse_dates=True)["annualized_percent"]
    schedule = an.build_schedule(returns.index)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 3: inference, robustness, tables and figures")
    print(f"  bootstrap        : stationary, expected block {BLOCK_LENGTH} days, "
          f"{N_BOOTSTRAP} replications, seed {SEED}")
    print(f"  cost levels      : {', '.join(f'{c} bp' for c in COST_LEVELS_BPS)} "
          f"(principal {PRINCIPAL_COST_BPS} bp)")

    # --- Task 3: run every cost level ---
    histories_by_cost, performance_by_cost, daily_by_cost, allocations_by_cost = {}, {}, {}, {}
    for cost in COST_LEVELS_BPS:
        h, m, d, a = run_cost_level(returns, rf_prev, schedule, cost)
        histories_by_cost[cost], performance_by_cost[cost] = h, m
        daily_by_cost[cost], allocations_by_cost[cost] = d, a

    principal_daily = daily_by_cost[PRINCIPAL_COST_BPS]
    principal_metrics = performance_by_cost[PRINCIPAL_COST_BPS]
    allocations = allocations_by_cost[PRINCIPAL_COST_BPS]

    # The selection rule uses gross candidate returns, so it must be identical
    # at every cost level. Verified, not assumed.
    reference = [r["selected"] for r in allocations_by_cost[PRINCIPAL_COST_BPS]]
    for cost in COST_LEVELS_BPS:
        if [r["selected"] for r in allocations_by_cost[cost]] != reference:
            raise AssertionError(f"selection differs at {cost} bp; it must not")
    print(f"  selection        : identical at all cost levels "
          f"({len(reference)} decisions)")

    # --- Task 2: paired inference at every cost level ---
    inference = []
    for cost in COST_LEVELS_BPS:
        daily = daily_by_cost[cost]
        testable = {
            name: (daily[DYNAMIC] - daily[name]).to_numpy()
            for name in BENCHMARKS
            if not np.array_equal(daily[DYNAMIC].to_numpy(), daily[name].to_numpy())
        }
        boot = stationary_bootstrap_means(testable) if testable else {}
        for name in BENCHMARKS:
            record = paired_comparison(daily[DYNAMIC], daily[name],
                                       boot_means=boot.get(name))
            record["cost_bps"] = cost
            record["benchmark"] = name
            inference.append(record)
        print(f"  bootstrap {cost:>2} bp : {len(testable)} testable comparisons, "
              f"{len(BENCHMARKS) - len(testable)} identical")

    # Supplementary multiplicity note. Three testable comparisons are made per
    # cost level, so a nominal 5% threshold overstates the evidence from the
    # smallest p-value. The protocol did not prespecify a correction, so the
    # prespecified p-values stand as the reported test and this Bonferroni
    # adjustment is recorded alongside them as context, not as a substitute.
    n_testable = sum(1 for r in inference
                     if r["status"] == "tested"
                     and r["cost_bps"] == PRINCIPAL_COST_BPS)
    for record in inference:
        record["n_comparisons_at_cost_level"] = n_testable
        record["bonferroni_p_value"] = (
            min(1.0, record["p_value"] * n_testable)
            if record["status"] == "tested" else np.nan
        )

    inference_frame = pd.DataFrame(inference)[
        ["cost_bps", "benchmark", "status", "n_days", "n_days_differing",
         "max_abs_daily_difference", "mean_daily_difference",
         "annualized_difference", "ci_low", "ci_high", "information_ratio",
         "p_value", "p_value_resolution", "n_comparisons_at_cost_level",
         "bonferroni_p_value"]
    ]

    # --- Task 4: selection behaviour ---
    frequency, advantage, margin_summary = selection_summary(allocations)

    # --- Tasks 5 and 6: tables and figures ---
    panel_a, panel_b, notes = descriptive_table(returns, risk_free_raw,
                                                principal_daily.index)
    performance = performance_table(principal_metrics)
    inference_panel_a, inference_panel_b = inference_table(inference,
                                                           performance_by_cost)

    panel_a.to_csv(RESULTS_DIR / "table1_descriptive_statistics.csv")
    panel_b.to_csv(RESULTS_DIR / "table1b_correlations.csv")
    notes.to_csv(RESULTS_DIR / "table1c_sample_notes.csv", index=False)
    performance.to_csv(RESULTS_DIR / "table2_performance.csv")
    inference_panel_a.to_csv(RESULTS_DIR / "table3a_paired_comparisons.csv", index=False)
    inference_panel_b.to_csv(RESULTS_DIR / "table3b_cost_robustness.csv")

    write_latex(panel_a, RESULTS_DIR / "table1_descriptive_statistics.tex",
                caption="Descriptive statistics for the four instruments, daily "
                        "adjusted total returns.",
                label="tab:descriptive",
                note="Daily simple returns from adjusted total-return prices. "
                     "Annualised figures use 252 trading days. ES 95\\% is the "
                     "mean of the worst 5\\% of daily returns.")
    write_latex(panel_b, RESULTS_DIR / "table1b_correlations.tex",
                caption="Correlations of daily returns.", label="tab:correlations")
    write_latex(performance, RESULTS_DIR / "table2_performance.tex",
                caption="Out-of-sample portfolio performance, 10 basis points per "
                        "dollar traded.",
                label="tab:performance",
                note="Net of transaction costs. The arithmetic annualised mean "
                     "net return and the compounded annual growth rate (CAGR) "
                     "are reported separately and are not interchangeable. "
                     "Maximum drawdown is measured on the net wealth path "
                     "including the initial wealth of 1. Dynamic ES is "
                     "numerically identical to static gold, because the rule "
                     "selected gold at all 201 decisions.")
    write_latex(inference_panel_a, RESULTS_DIR / "table3a_paired_comparisons.tex",
                caption="Paired daily net-return differences, dynamic strategy "
                        "minus each benchmark.",
                label="tab:inference", index=False,
                note="H0: equal expected daily net returns; two-sided. "
                     "Inference by stationary block bootstrap, expected block "
                     "length 21 trading days, 10{,}000 replications, seed "
                     "20261025. Differences are annualised and expressed in "
                     "percentage points. The dynamic strategy and static gold "
                     "are the same series, so that comparison has no sampling "
                     "variation and no p-value is reported. Three testable "
                     "comparisons are made per cost level; a Bonferroni-adjusted "
                     "p-value is reported in bootstrap\_inference.csv as "
                     "context. The correction was not prespecified, so the "
                     "p-values above are the prespecified test.")
    write_latex(inference_panel_b, RESULTS_DIR / "table3b_cost_robustness.tex",
                caption="Transaction-cost robustness: annualised mean net return "
                        "and Sharpe ratio at 10, 25 and 50 basis points.",
                label="tab:robustness")

    inference_frame.to_csv(RESULTS_DIR / "bootstrap_inference.csv", index=False)
    frequency.to_csv(RESULTS_DIR / "selection_frequency.csv", index=False)
    advantage.to_csv(RESULTS_DIR / "es_advantage_history.csv", index=False)
    margin_summary.to_csv(RESULTS_DIR / "es_advantage_summary.csv", index=False)
    pd.concat(
        {f"{c}bp": performance_by_cost[c] for c in COST_LEVELS_BPS},
        names=["cost_level", "strategy"],
    ).to_csv(RESULTS_DIR / "performance_all_cost_levels.csv")
    for cost in COST_LEVELS_BPS:
        daily_by_cost[cost].to_csv(
            RESULTS_DIR / f"daily_portfolio_returns_{cost}bp.csv"
        )

    figure_cumulative_wealth(principal_daily, RESULTS_DIR / "figure1_cumulative_wealth")
    figure_es_differences(advantage, RESULTS_DIR / "figure2_es_differences")

    # --- Report ---
    print()
    print("Task 4 - candidate selection frequency")
    print(frequency.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("Expected Shortfall margin of the selected candidate (bp/day)")
    print(margin_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"Task 5/2 - performance at {PRINCIPAL_COST_BPS} bp")
    print(performance.to_string(float_format=lambda v: f"{v:10.4f}"))
    print()
    print("Task 2 - paired comparisons (annualised pp, two-sided)")
    print(inference_panel_a.to_string(index=False))
    print()
    print(f"  Multiplicity note: {n_testable} testable comparisons per cost level. "
          "Bonferroni-adjusted p-values (supplementary, not prespecified):")
    for record in inference:
        if record["status"] == "tested":
            print(f"    {record['cost_bps']:>2} bp  "
                  f"{STRATEGY_LABELS[record['benchmark']]:<22} "
                  f"nominal p={record['p_value']:.4f}  "
                  f"Bonferroni p={record['bonferroni_p_value']:.4f}")
    print()
    print("Task 3 - cost robustness (annualised mean net return, %)")
    print(inference_panel_b.to_string(float_format=lambda v: f"{v:9.4f}"))
    print()
    print(f"Results written to {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

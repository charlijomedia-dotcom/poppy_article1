"""Phase 2: portfolio simulation, Expected Shortfall selection and core metrics.

Run:  python3 analysis.py

Reads the frozen Phase 1 outputs (no downloads) and writes
data/results/performance_metrics.csv,
data/results/daily_portfolio_returns.csv and
data/results/dynamic_allocation_history.csv.

Implements research_protocol.md as frozen, decisions D1-D10. Every strategy runs
through one simulation engine (`simulate`), so the decision/execution timing,
the weight drift and the transaction-cost treatment are identical by
construction rather than by coincidence.

Timing within a day (decision D2). The portfolio enters day t holding the
weights set at the close of t-1 and earns that allocation's return during day t.
Weights then drift with relative returns. If t is an execution date, the
portfolio trades from the drifted weights to the new target at that day's close
and pays the cost, so the new weights first earn a return on t+1. A decision
taken at the close of the last trading day of month M is executed at the close
of the first trading day of month M+1, so nothing is traded at a price that was
unavailable when the decision was made.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Protocol parameters (research_protocol.md sections 5-8) ------------------

ASSETS = ["SPY", "AGG", "GLD", "SLV"]

# Candidate portfolios (section 5). The 5% metal sleeve is funded pro rata from
# equity and bonds, preserving the 60:40 ratio within the non-metal part.
CANDIDATES = {
    "no_metal": {"SPY": 0.60, "AGG": 0.40},
    "gold": {"SPY": 0.57, "AGG": 0.38, "GLD": 0.05},
    "silver": {"SPY": 0.57, "AGG": 0.38, "SLV": 0.05},
}

ESTIMATION_WINDOW = 756          # trading days (section 6)
ES_TAIL = 0.05                   # 95% Expected Shortfall
COST_PER_DOLLAR = 0.0010         # 10 basis points, principal assumption (section 7)
VOL_TARGET_ANNUAL = 0.10         # volatility-managed benchmark target (D4)
VOL_WINDOW = 63                  # trading days for the volatility estimate (D4)
MAX_RISKY_WEIGHT = 1.0           # no leverage (D4)
TRADING_DAYS_PER_YEAR = 252

RETURNS_PATH = Path("data/processed/daily_returns.csv")
RISK_FREE_PATH = Path("data/processed/risk_free_daily.csv")
RESULTS_DIR = Path("data/results")


def load_inputs(returns_path=RETURNS_PATH, risk_free_path=RISK_FREE_PATH):
    """Load the frozen Phase 1 dataset. Never downloads anything.

    Returns the asset return panel and the *lagged* daily risk-free rate: the
    yield published on t-1 is what the cash leg earns on day t (decision D3).
    The input file is dated to the publication day, so the lag is applied here,
    once, and `rf_prev` is the only risk-free series the rest of the module sees.
    """
    for path in (returns_path, risk_free_path):
        if not path.exists():
            raise SystemExit(
                f"required Phase 1 input {path} is missing. Run "
                "'python3 download_data.py' to rebuild it from the frozen raw data."
            )

    returns = pd.read_csv(returns_path, index_col="Date", parse_dates=True)
    returns = returns[ASSETS].sort_index()

    risk_free = pd.read_csv(risk_free_path, index_col="Date", parse_dates=True).sort_index()
    if not risk_free.index.equals(returns.index):
        raise SystemExit("risk-free series and return panel do not share an index")

    rf_prev = risk_free["daily_rate"].shift(1)
    rf_prev.name = "rf_prev"
    return returns, rf_prev


def weights_vector(mapping):
    """Turn a {ticker: weight} mapping into an array ordered like ASSETS."""
    return np.array([float(mapping.get(a, 0.0)) for a in ASSETS])


def month_end_dates(index):
    """Last trading day of each month present in the index."""
    frame = pd.Series(index, index=index)
    return pd.DatetimeIndex(frame.groupby([index.year, index.month]).last().values)


def month_start_dates(index):
    """First trading day of each month present in the index."""
    frame = pd.Series(index, index=index)
    return pd.DatetimeIndex(frame.groupby([index.year, index.month]).first().values)


def build_schedule(index, estimation_window=ESTIMATION_WINDOW):
    """Pair each eligible month-end decision date with its execution date.

    A decision date is eligible only once a *complete* estimation window of
    returns is available up to and including that date, so the first allocation
    is never based on a partial window (decision D1).
    """
    positions = {d: i for i, d in enumerate(index)}
    schedule = []
    for decision in month_end_dates(index):
        # Number of return observations up to and including the decision date.
        if positions[decision] + 1 < estimation_window:
            continue
        nxt = positions[decision] + 1
        if nxt >= len(index):
            continue  # no execution date left in the sample
        schedule.append((decision, index[nxt]))
    return schedule


def monthly_rebalanced_gross(returns, target, rebalance_dates):
    """Gross daily returns of a fixed-target portfolio restored monthly.

    Used for the Expected Shortfall estimation input (decision D5) and for the
    volatility estimate of the managed benchmark (D4), so both are measured on
    the same monthly-rebalanced basis as the strategy actually traded.

    The value at date t depends only on returns up to and including t, so
    computing the whole series once and slicing trailing windows from it
    introduces no look-ahead.
    """
    rebalance = set(pd.DatetimeIndex(rebalance_dates))
    weights = target.copy()
    out = np.empty(len(returns))
    for i, (date, row) in enumerate(zip(returns.index, returns.to_numpy())):
        out[i] = weights @ row
        value = weights * (1.0 + row)
        weights = value / value.sum()
        if date in rebalance:
            weights = target.copy()
    return pd.Series(out, index=returns.index, name="gross")


def expected_shortfall(series, tail=ES_TAIL):
    """Historical Expected Shortfall: mean of the worst floor(tail * n) returns.

    Returned as a negative number for a loss, on the same scale as the returns.
    """
    values = np.sort(np.asarray(series, dtype=float))
    k = int(np.floor(tail * len(values)))
    if k < 1:
        raise ValueError(f"series of length {len(values)} is too short for a {tail:.0%} tail")
    return float(values[:k].mean())


def simulate(returns, rf_prev, schedule, target_for, cost_per_dollar=COST_PER_DOLLAR):
    """Run one strategy and return its daily net returns and weight history.

    `target_for` maps a decision date to (risky weight vector, cash weight). It
    is called only with decision dates, and the resulting target is applied at
    the close of the paired execution date, so a target may use no information
    after its decision date.

    The portfolio starts fully in cash on the first execution date, earns the
    cash rate that day, and buys into its first allocation at that day's close.
    Every strategy is charged on the same basis: the cost is
    `cost_per_dollar` times the sum of absolute changes in the four risky
    weights. Cash bears no transaction cost, so the volatility-managed benchmark
    pays only for the securities it actually trades.
    """
    targets = {execution: target_for(decision) for decision, execution in schedule}
    first_execution = schedule[0][1]
    index = returns.index[returns.index >= first_execution]

    weights = np.zeros(len(ASSETS))
    cash = 1.0

    records = []
    for date in index:
        row = returns.loc[date].to_numpy()
        rf = float(rf_prev.loc[date])

        # Return earned during the day on the allocation held at its start.
        gross = float(weights @ row + cash * rf)

        # Positions drift with relative returns.
        value = weights * (1.0 + row)
        value_cash = cash * (1.0 + rf)
        total = value.sum() + value_cash
        weights_drifted = value / total
        cash_drifted = value_cash / total

        traded = 0.0
        if date in targets:
            target_weights, target_cash = targets[date]
            traded = float(np.abs(target_weights - weights_drifted).sum())
            weights, cash = target_weights.copy(), float(target_cash)
        else:
            weights, cash = weights_drifted, float(cash_drifted)

        # The cost is a fraction of portfolio value at the close, after the
        # day's return has accrued. Days without a trade are left exactly equal
        # to the gross return rather than round-tripped through the cost factor.
        cost = cost_per_dollar * traded
        net = (1.0 + gross) * (1.0 - cost) - 1.0 if cost else gross

        records.append(
            {
                "Date": date,
                "gross_return": gross,
                "net_return": net,
                "cost": (1.0 + gross) * cost,
                "traded_fraction": traded,
                "cash_weight": cash,
                **{f"w_{a}": w for a, w in zip(ASSETS, weights)},
            }
        )

    history = pd.DataFrame(records).set_index("Date")
    return history


def max_drawdown(net_returns):
    """Largest peak-to-trough decline of the cumulative net return series."""
    cumulative = (1.0 + net_returns).cumprod()
    return float((cumulative / cumulative.cummax() - 1.0).min())


def performance_metrics(history, rf_prev):
    """Summary indicators for one strategy (research_protocol.md section 8)."""
    net = history["net_return"]
    n = len(net)
    years = n / TRADING_DAYS_PER_YEAR
    excess = net - rf_prev.reindex(net.index)
    cumulative = float((1.0 + net).prod())
    excess_sd = float(excess.std(ddof=1))
    # One-way turnover (D6); costs are charged on the full traded fraction.
    one_way = history["traded_fraction"].sum() / 2.0

    return {
        "annualized_net_return": float(net.mean() * TRADING_DAYS_PER_YEAR),
        "cagr": cumulative ** (1.0 / years) - 1.0,
        "annualized_volatility": float(net.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe_ratio": (
            float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess_sd)
            if excess_sd > 0 else float("nan")
        ),
        "expected_shortfall_95": expected_shortfall(net),
        "max_drawdown": max_drawdown(net),
        "annualized_turnover_one_way": float(one_way / years),
        "total_cost_paid": float(history["cost"].sum()),
        "cumulative_net_return": cumulative - 1.0,
        "n_days": n,
        "years": years,
    }


def build_strategies(returns, schedule):
    """Define the five approved strategies as decision-date target functions.

    Returns the mapping of strategy name to target function, plus the per-date
    Expected Shortfall record of the dynamic rule.
    """
    rebalance_dates = month_start_dates(returns.index)
    candidate_weights = {name: weights_vector(w) for name, w in CANDIDATES.items()}
    # Gross monthly-rebalanced return series per candidate (D5). Causal: the
    # value at t uses only returns up to t.
    candidate_gross = {
        name: monthly_rebalanced_gross(returns, weights, rebalance_dates)
        for name, weights in candidate_weights.items()
    }

    allocation_records = []

    def dynamic_target(decision):
        """Lowest 95% Expected Shortfall over the trailing 756 observations."""
        shortfalls = {}
        for name, series in candidate_gross.items():
            window = series.loc[:decision].tail(ESTIMATION_WINDOW)
            if len(window) != ESTIMATION_WINDOW:
                raise AssertionError(
                    f"incomplete estimation window at {decision.date()}: "
                    f"{len(window)} of {ESTIMATION_WINDOW} observations"
                )
            shortfalls[name] = expected_shortfall(window)
        # Highest (least negative) ES is the smallest loss. Ties resolve to the
        # first candidate in CANDIDATES order, deterministically.
        selected = max(shortfalls, key=lambda name: shortfalls[name])
        allocation_records.append(
            {
                "decision_date": decision,
                "selected": selected,
                **{f"es_{name}": shortfalls[name] for name in CANDIDATES},
            }
        )
        return candidate_weights[selected].copy(), 0.0

    def static_target(name):
        def target(decision):
            return candidate_weights[name].copy(), 0.0
        return target

    def volatility_managed_target(decision):
        """10% annualised target from the preceding 63 days, no leverage (D4).

        The protocol specifies data *strictly before* the decision date, so the
        window ends on the previous trading day.
        """
        history = candidate_gross["no_metal"].loc[:decision].iloc[:-1]
        window = history.tail(VOL_WINDOW)
        if len(window) != VOL_WINDOW:
            raise AssertionError(
                f"incomplete volatility window at {decision.date()}: "
                f"{len(window)} of {VOL_WINDOW} observations"
            )
        sigma = float(window.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        risky = min(MAX_RISKY_WEIGHT, VOL_TARGET_ANNUAL / sigma)
        return candidate_weights["no_metal"] * risky, 1.0 - risky

    strategies = {
        "traditional_60_40": static_target("no_metal"),
        "static_gold": static_target("gold"),
        "static_silver": static_target("silver"),
        "dynamic_es": dynamic_target,
        "volatility_managed_60_40": volatility_managed_target,
    }
    return strategies, allocation_records


def main():
    returns, rf_prev = load_inputs()
    schedule = build_schedule(returns.index)
    if not schedule:
        raise SystemExit("no eligible decision dates: sample shorter than the estimation window")

    strategies, allocation_records = build_strategies(returns, schedule)

    histories = {}
    for name, target_for in strategies.items():
        histories[name] = simulate(returns, rf_prev, schedule, target_for)

    metrics = pd.DataFrame(
        {name: performance_metrics(h, rf_prev) for name, h in histories.items()}
    ).T
    metrics.index.name = "strategy"

    daily = pd.DataFrame({name: h["net_return"] for name, h in histories.items()})
    daily.index.name = "Date"

    allocations = pd.DataFrame(allocation_records)
    execution_of = {decision: execution for decision, execution in schedule}
    allocations["execution_date"] = allocations["decision_date"].map(execution_of)
    allocations = allocations[
        ["decision_date", "execution_date", "selected"]
        + [f"es_{name}" for name in CANDIDATES]
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "performance_metrics.csv")
    daily.to_csv(RESULTS_DIR / "daily_portfolio_returns.csv")
    allocations.to_csv(RESULTS_DIR / "dynamic_allocation_history.csv", index=False)

    first_decision, first_execution = schedule[0]
    print("Phase 2 out-of-sample simulation")
    print(f"  transaction cost      : {COST_PER_DOLLAR * 10000:.0f} bp per dollar traded")
    print(f"  estimation window     : {ESTIMATION_WINDOW} trading days")
    print(f"  first decision date   : {first_decision.date()}")
    print(f"  first execution date  : {first_execution.date()}")
    print(f"  evaluation period     : {daily.index.min().date()} to {daily.index.max().date()}"
          f"  ({len(daily)} trading days, {len(daily)/TRADING_DAYS_PER_YEAR:.1f} years)")
    print(f"  monthly decisions     : {len(schedule)}")
    print()
    shown = [
        "annualized_net_return", "cagr", "sharpe_ratio",
        "expected_shortfall_95", "max_drawdown", "annualized_turnover_one_way",
    ]
    print(metrics[shown].to_string(float_format=lambda v: f"{v:11.6f}"))
    print()
    counts = allocations["selected"].value_counts()
    print("Dynamic rule selections (of "
          f"{len(allocations)} monthly decisions): "
          + ", ".join(f"{name}={counts.get(name, 0)}" for name in CANDIDATES))
    print(f"\nResults written to {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

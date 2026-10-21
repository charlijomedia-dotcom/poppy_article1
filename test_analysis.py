"""Phase 2 tests: portfolio mechanics, cost accounting and look-ahead safety.

Two kinds of test:

* **Synthetic** — tiny hand-computable examples that pin down drift,
  rebalancing, transaction costs, the risk-free lag and Expected Shortfall
  against numbers worked out by hand in the test itself. These synthetic series
  are test fixtures only and are never used as research data.
* **Frozen dataset** — invariants checked on the real Phase 1 data: weights sum
  to one, no short positions, no leverage, consistent cost treatment across
  strategies, and that no future information can reach an earlier return.

Run:  python3 -m pytest test_analysis.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis as an

RESULTS_DIR = an.RESULTS_DIR
WEIGHT_COLS = [f"w_{a}" for a in an.ASSETS]


# --- Synthetic fixtures (test-only; never research data) ---------------------

def _dates(n, start="2020-01-01"):
    return pd.bdate_range(start=start, periods=n)


def _returns(rows, dates):
    """Build a return panel from rows of [SPY, AGG, GLD, SLV]."""
    return pd.DataFrame(rows, index=dates, columns=an.ASSETS, dtype=float)


def _flat_rf(dates, value=0.0):
    return pd.Series(value, index=dates, name="rf_prev", dtype=float)


def _constant_target(mapping, cash=0.0):
    vector = an.weights_vector(mapping)
    return lambda decision: (vector.copy(), cash)


# --- Expected Shortfall ------------------------------------------------------

def test_expected_shortfall_is_the_mean_of_the_worst_floor_tail():
    # 100 observations, tail 5% -> floor(5.0) = 5 worst. Worst five of
    # 0.00, 0.01, ... 0.99 are 0.00..0.04, mean 0.02.
    series = pd.Series(np.arange(100) / 100.0)
    assert an.expected_shortfall(series, tail=0.05) == pytest.approx(0.02)


def test_expected_shortfall_uses_37_observations_for_the_protocol_window():
    # research_protocol.md section 6 pins k = floor(0.05 * 756) = 37.
    assert int(np.floor(0.05 * an.ESTIMATION_WINDOW)) == 37
    series = pd.Series(np.arange(an.ESTIMATION_WINDOW, dtype=float))
    # Worst 37 of 0..755 are 0..36, mean 18.0.
    assert an.expected_shortfall(series) == pytest.approx(18.0)


def test_expected_shortfall_is_negative_for_a_loss_tail():
    series = pd.Series([-0.05] * 5 + [0.01] * 95)
    assert an.expected_shortfall(series, tail=0.05) == pytest.approx(-0.05)


def test_expected_shortfall_rejects_a_series_too_short_for_the_tail():
    with pytest.raises(ValueError):
        an.expected_shortfall(pd.Series([0.01, 0.02]), tail=0.05)


# --- Drift and monthly rebalancing ------------------------------------------

def test_monthly_rebalanced_series_drifts_between_rebalances():
    """Hand-computed: 50/50, SPY +10% on day 1, nothing else.

    Day 1 gross = 0.5 * 0.10 = 0.05. Weights drift to 0.55/1.05 and 0.50/1.05
    = 0.523809.../0.476190... With no rebalance, day 3's +10% on SPY gives
    0.5238095 * 0.10 = 0.05238095.
    """
    dates = _dates(3)
    returns = _returns([[0.10, 0, 0, 0], [0, 0, 0, 0], [0.10, 0, 0, 0]], dates)
    target = an.weights_vector({"SPY": 0.5, "AGG": 0.5})

    drifted = an.monthly_rebalanced_gross(returns, target, rebalance_dates=[])
    assert drifted.iloc[0] == pytest.approx(0.05)
    assert drifted.iloc[1] == pytest.approx(0.0)
    assert drifted.iloc[2] == pytest.approx(0.55 / 1.05 * 0.10)


def test_monthly_rebalanced_series_restores_target_on_rebalance_dates():
    """Same path, but rebalancing on day 2 resets weights to 50/50.

    Day 3's +10% on SPY then gives exactly 0.5 * 0.10 = 0.05.
    """
    dates = _dates(3)
    returns = _returns([[0.10, 0, 0, 0], [0, 0, 0, 0], [0.10, 0, 0, 0]], dates)
    target = an.weights_vector({"SPY": 0.5, "AGG": 0.5})

    rebalanced = an.monthly_rebalanced_gross(returns, target, rebalance_dates=[dates[1]])
    assert rebalanced.iloc[2] == pytest.approx(0.05)


# --- Transaction costs and net returns --------------------------------------

def test_initial_purchase_and_drift_costs_are_hand_computable():
    """Three days, 60/40, 10 bp, zero risk-free rate.

    Day 1 (first execution): starts in cash, gross 0, buys in. Traded fraction
      |0.6-0| + |0.4-0| = 1.0, so cost = 0.0010 and net = -0.0010.
    Day 2: SPY +10%. gross = 0.6 * 0.10 = 0.06, no trade, net = 0.06.
      Weights drift to 0.66/1.06 and 0.40/1.06.
    Day 3 (execution): zero returns, so the trade back to 60/40 is
      2 * |0.6 - 0.66/1.06| = 0.0452830..., cost = 4.528302e-5,
      net = -4.528302e-5.
    """
    dates = _dates(3)
    returns = _returns([[0, 0, 0, 0], [0.10, 0, 0, 0], [0, 0, 0, 0]], dates)
    rf = _flat_rf(dates)
    schedule = [(dates[0], dates[0]), (dates[1], dates[2])]

    history = an.simulate(returns, rf, schedule,
                          _constant_target({"SPY": 0.60, "AGG": 0.40}),
                          cost_per_dollar=0.0010)

    assert history["traded_fraction"].iloc[0] == pytest.approx(1.0)
    assert history["net_return"].iloc[0] == pytest.approx(-0.0010)

    assert history["traded_fraction"].iloc[1] == pytest.approx(0.0)
    assert history["net_return"].iloc[1] == pytest.approx(0.06)

    expected_traded = 2 * abs(0.60 - 0.66 / 1.06)
    assert history["traded_fraction"].iloc[2] == pytest.approx(expected_traded)
    assert history["net_return"].iloc[2] == pytest.approx(-0.0010 * expected_traded)


def test_net_return_equals_gross_compounded_with_the_cost():
    """net = (1 + gross) * (1 - cost_rate * traded) - 1, exactly."""
    dates = _dates(4)
    returns = _returns([[0.02, 0.01, 0, 0], [-0.03, 0.005, 0, 0],
                        [0.01, -0.002, 0, 0], [0.004, 0.001, 0, 0]], dates)
    rf = _flat_rf(dates, 0.0001)
    schedule = [(dates[0], dates[0]), (dates[1], dates[2])]
    history = an.simulate(returns, rf, schedule,
                          _constant_target({"SPY": 0.60, "AGG": 0.40}),
                          cost_per_dollar=0.0025)

    traded = history["traded_fraction"] > 0
    expected = (1 + history.loc[traded, "gross_return"]) * (
        1 - 0.0025 * history.loc[traded, "traded_fraction"]
    ) - 1
    assert traded.any()
    assert np.allclose(history.loc[traded, "net_return"], expected, atol=0, rtol=1e-15)
    # Days without a trade carry no cost at all.
    assert np.array_equal(history.loc[~traded, "net_return"].to_numpy(),
                          history.loc[~traded, "gross_return"].to_numpy())


def test_zero_cost_makes_net_equal_gross():
    dates = _dates(5)
    rng = np.random.default_rng(0)  # test fixture only
    returns = _returns(rng.normal(0, 0.01, size=(5, 4)), dates)
    rf = _flat_rf(dates)
    schedule = [(dates[0], dates[0]), (dates[2], dates[3])]
    history = an.simulate(returns, rf, schedule,
                          _constant_target({"SPY": 0.60, "AGG": 0.40}),
                          cost_per_dollar=0.0)
    assert np.array_equal(history["net_return"].to_numpy(),
                          history["gross_return"].to_numpy())


def test_cash_bears_no_transaction_cost():
    """A strategy that buys only 50% risky pays half the full-investment cost."""
    dates = _dates(2)
    returns = _returns([[0, 0, 0, 0], [0, 0, 0, 0]], dates)
    rf = _flat_rf(dates)
    schedule = [(dates[0], dates[0])]
    half = an.simulate(returns, rf, schedule,
                       _constant_target({"SPY": 0.30, "AGG": 0.20}, cash=0.5),
                       cost_per_dollar=0.0010)
    assert half["traded_fraction"].iloc[0] == pytest.approx(0.5)
    assert half["net_return"].iloc[0] == pytest.approx(-0.0005)


# --- Risk-free lag (no same-date or future yields) --------------------------

def test_load_inputs_lags_the_risk_free_rate(tmp_path):
    """The yield published on t-1 must be what day t earns (decision D3)."""
    dates = _dates(5)
    returns = _returns(np.zeros((5, 4)), dates)
    returns.index.name = "Date"
    rates = pd.DataFrame(
        {
            "annualized_percent": [1.0, 2.0, 3.0, 4.0, 5.0],
            "daily_rate": [0.001, 0.002, 0.003, 0.004, 0.005],
            "carried_forward": 0,
        },
        index=dates,
    )
    rates.index.name = "Date"
    returns_path, rates_path = tmp_path / "r.csv", tmp_path / "rf.csv"
    returns.to_csv(returns_path)
    rates.to_csv(rates_path)

    _, rf_prev = an.load_inputs(returns_path, rates_path)

    assert np.isnan(rf_prev.iloc[0]), "the first day has no previous yield"
    for i in range(1, len(dates)):
        assert rf_prev.iloc[i] == pytest.approx(rates["daily_rate"].iloc[i - 1])
        assert rf_prev.iloc[i] != pytest.approx(rates["daily_rate"].iloc[i]), (
            "day t must not earn the yield dated t"
        )


def test_cash_only_portfolio_earns_the_previous_days_yield():
    """A 100% cash portfolio's return on day t must equal the yield from t-1."""
    dates = _dates(4)
    returns = _returns(np.zeros((4, 4)), dates)
    rf_raw = pd.Series([0.001, 0.002, 0.003, 0.004], index=dates)
    rf_prev = rf_raw.shift(1)
    schedule = [(dates[0], dates[1])]

    history = an.simulate(returns, rf_prev, schedule,
                          _constant_target({}, cash=1.0), cost_per_dollar=0.0010)
    # Evaluation starts on the execution date, dates[1].
    for date in history.index:
        assert history.loc[date, "gross_return"] == pytest.approx(rf_prev.loc[date])
        assert history.loc[date, "gross_return"] != pytest.approx(rf_raw.loc[date]), (
            f"{date.date()} used the same-date yield"
        )


def test_future_risk_free_changes_cannot_affect_earlier_returns():
    """Perturbing the yield after a cut-off must leave earlier returns identical."""
    dates = _dates(10)
    returns = _returns(np.full((10, 4), 0.001), dates)
    rf_raw = pd.Series(0.0001, index=dates)
    schedule = [(dates[0], dates[1]), (dates[4], dates[5])]
    # Half in cash, so the risk-free rate genuinely enters the return.
    target = _constant_target({"SPY": 0.30, "AGG": 0.20}, cash=0.5)

    base = an.simulate(returns, rf_raw.shift(1), schedule, target)

    bumped = rf_raw.copy()
    bumped.iloc[7:] = 0.05  # implausible future yield
    perturbed = an.simulate(returns, bumped.shift(1), schedule, target)

    cutoff = dates[7]
    before = base.index < cutoff
    assert np.array_equal(
        base.loc[before, "net_return"].to_numpy(),
        perturbed.loc[before, "net_return"].to_numpy(),
    )
    # Sanity: the perturbation does change something later on.
    assert not np.array_equal(base["net_return"].to_numpy(),
                              perturbed["net_return"].to_numpy())


# Floating-point reproducibility floor. Repeating a run over an identical copy
# of the frozen data already moves a few daily returns by about 2e-16, because
# pandas row extraction and dot products are not bit-reproducible. Genuine
# look-ahead from the perturbations below would move returns by order 1e-3, so
# this threshold separates the two by eleven orders of magnitude.
FLOAT_NOISE_FLOOR = 1e-12


def test_repeating_a_run_on_identical_data_is_reproducible_to_the_noise_floor():
    """Establishes the floor the look-ahead tests below are measured against."""
    returns, rf_prev = an.load_inputs()
    schedule = an.build_schedule(returns.index)
    first, _ = an.build_strategies(returns, schedule)
    second, _ = an.build_strategies(returns.copy(), schedule)
    a = an.simulate(returns, rf_prev, schedule, first["dynamic_es"])["net_return"]
    b = an.simulate(returns.copy(), rf_prev, schedule, second["dynamic_es"])["net_return"]
    assert np.abs(a.to_numpy() - b.to_numpy()).max() < FLOAT_NOISE_FLOOR


# --- Decision and execution timing ------------------------------------------

def test_new_target_first_earns_a_return_the_day_after_execution():
    """A target decided at M-end is executed at the next close (decision D2).

    The execution day itself must still earn the *old* allocation's return;
    the new weights can only affect the following day.
    """
    dates = _dates(4)
    # SPY and AGG move in opposite directions on the execution day, so holding
    # the old versus the new allocation gives visibly different returns.
    returns = _returns([[0, 0, 0, 0], [0, 0, 0, 0],
                        [0.10, -0.10, 0, 0], [0.10, -0.10, 0, 0]], dates)
    rf = _flat_rf(dates)

    # Start 100% AGG, then switch to 100% SPY, executed at dates[2].
    def target_for(decision):
        if decision == dates[0]:
            return an.weights_vector({"AGG": 1.0}), 0.0
        return an.weights_vector({"SPY": 1.0}), 0.0

    schedule = [(dates[0], dates[1]), (dates[1], dates[2])]
    history = an.simulate(returns, rf, schedule, target_for, cost_per_dollar=0.0)

    # dates[2] is the execution day: still fully in AGG, so -10%.
    assert history.loc[dates[2], "gross_return"] == pytest.approx(-0.10)
    # dates[3]: the new allocation is live, so +10%.
    assert history.loc[dates[3], "gross_return"] == pytest.approx(0.10)


def test_evaluation_starts_on_the_first_execution_date():
    dates = _dates(6)
    returns = _returns(np.zeros((6, 4)), dates)
    rf = _flat_rf(dates)
    schedule = [(dates[2], dates[3])]
    history = an.simulate(returns, rf, schedule,
                          _constant_target({"SPY": 0.60, "AGG": 0.40}))
    assert history.index.min() == dates[3]
    assert history.index.max() == dates[-1]


# --- Frozen dataset: schedule and estimation window -------------------------

@pytest.fixture(scope="module")
def frozen():
    returns, rf_prev = an.load_inputs()
    schedule = an.build_schedule(returns.index)
    strategies, allocations = an.build_strategies(returns, schedule)
    histories = {
        name: an.simulate(returns, rf_prev, schedule, fn)
        for name, fn in strategies.items()
    }
    return returns, rf_prev, schedule, histories, allocations


def test_first_decision_has_a_complete_estimation_window(frozen):
    returns, _, schedule, _, _ = frozen
    first_decision = schedule[0][0]
    available = returns.loc[:first_decision]
    assert len(available) >= an.ESTIMATION_WINDOW
    # And the previous month-end did not have a full window, so the schedule
    # starts as early as the protocol allows and no earlier.
    month_ends = an.month_end_dates(returns.index)
    earlier = month_ends[month_ends < first_decision]
    assert len(returns.loc[:earlier[-1]]) < an.ESTIMATION_WINDOW


def test_evaluation_begins_in_january_2010_per_decision_d1(frozen):
    _, _, schedule, _, _ = frozen
    first_decision, first_execution = schedule[0]
    assert str(first_decision.date()) == "2009-12-31"
    assert (first_execution.year, first_execution.month) == (2010, 1)


def test_every_dynamic_decision_used_a_full_window(frozen):
    returns, _, schedule, _, allocations = frozen
    assert len(allocations) == len(schedule)
    for record in allocations:
        window = returns.loc[:record["decision_date"]]
        assert len(window) >= an.ESTIMATION_WINDOW


def test_execution_always_follows_the_decision_date(frozen):
    returns, _, schedule, _, _ = frozen
    positions = {d: i for i, d in enumerate(returns.index)}
    for decision, execution in schedule:
        assert execution > decision
        assert positions[execution] == positions[decision] + 1


# --- Frozen dataset: weight invariants --------------------------------------

@pytest.mark.parametrize("name", [
    "traditional_60_40", "static_gold", "static_silver",
    "dynamic_es", "volatility_managed_60_40",
])
def test_weights_are_valid_every_day(frozen, name):
    _, _, _, histories, _ = frozen
    history = histories[name]
    weights = history[WEIGHT_COLS]

    # Non-negative: no short positions.
    assert (weights >= -1e-12).all().all()
    assert (history["cash_weight"] >= -1e-12).all()
    # No leverage: risky weights never exceed one.
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
    # Weights plus cash sum to one.
    total = weights.sum(axis=1) + history["cash_weight"]
    assert np.allclose(total, 1.0, atol=1e-12)


def test_fully_invested_strategies_hold_no_cash(frozen):
    _, _, _, histories, _ = frozen
    for name in ["traditional_60_40", "static_gold", "static_silver", "dynamic_es"]:
        assert np.allclose(histories[name]["cash_weight"], 0.0, atol=1e-12)


def test_volatility_managed_respects_the_target_and_the_cap(frozen):
    _, _, _, histories, _ = frozen
    risky = histories["volatility_managed_60_40"][WEIGHT_COLS].sum(axis=1)
    assert risky.max() <= an.MAX_RISKY_WEIGHT + 1e-9
    assert risky.min() > 0.0
    # It must actually de-risk sometimes, or the benchmark is just 60/40.
    assert risky.min() < 0.99


def test_static_strategies_hold_their_protocol_weights_at_execution(frozen):
    _, _, schedule, histories, _ = frozen
    executions = [e for _, e in schedule]
    for name, candidate in [("traditional_60_40", "no_metal"),
                            ("static_gold", "gold"),
                            ("static_silver", "silver")]:
        expected = an.weights_vector(an.CANDIDATES[candidate])
        observed = histories[name].loc[executions, WEIGHT_COLS].to_numpy()
        assert np.allclose(observed, expected, atol=1e-12)


# --- Frozen dataset: consistent cost treatment ------------------------------

def test_all_strategies_are_charged_the_same_rate_per_dollar_traded(frozen):
    """Static and dynamic strategies must share one cost rule (decision D9)."""
    _, _, _, histories, _ = frozen
    for name, history in histories.items():
        traded = history["traded_fraction"]
        active = traded > 1e-15
        implied = history.loc[active, "cost"] / (
            (1 + history.loc[active, "gross_return"]) * traded[active]
        )
        assert np.allclose(implied, an.COST_PER_DOLLAR, rtol=1e-12), name


def test_every_strategy_trades_on_exactly_the_execution_dates(frozen):
    _, _, schedule, histories, _ = frozen
    executions = pd.DatetimeIndex([e for _, e in schedule])
    for name, history in histories.items():
        traded_on = history.index[history["traded_fraction"] > 1e-15]
        assert set(traded_on) <= set(executions), name
        non_execution = history.index.difference(executions)
        assert (history.loc[non_execution, "traded_fraction"] == 0).all(), name


def test_all_strategies_share_the_same_evaluation_index(frozen):
    _, _, _, histories, _ = frozen
    indexes = [h.index for h in histories.values()]
    for index in indexes[1:]:
        assert index.equals(indexes[0])


def test_costs_reduce_returns(frozen):
    """Net must never exceed gross, and must be strictly below it when trading."""
    _, _, _, histories, _ = frozen
    for name, history in histories.items():
        assert (history["net_return"] <= history["gross_return"] + 1e-15).all(), name
        traded = history["traded_fraction"] > 1e-15
        assert (history.loc[traded, "net_return"]
                < history.loc[traded, "gross_return"]).all(), name


# --- Frozen dataset: look-ahead safety on the real pipeline -----------------

@pytest.mark.parametrize("name", ["dynamic_es", "volatility_managed_60_40"])
def test_future_returns_cannot_affect_earlier_net_returns(name):
    """Perturb asset returns after a cut-off; earlier results must be identical.

    This exercises the full estimation path, so it fails if the Expected
    Shortfall window or the volatility window ever reaches past its decision
    date.
    """
    returns, rf_prev = an.load_inputs()
    schedule = an.build_schedule(returns.index)
    cutoff = pd.Timestamp("2018-01-02")

    base_strategies, _ = an.build_strategies(returns, schedule)
    base = an.simulate(returns, rf_prev, schedule, base_strategies[name])

    perturbed_returns = returns.copy()
    perturbed_returns.loc[perturbed_returns.index >= cutoff] *= -3.0
    perturbed_strategies, _ = an.build_strategies(perturbed_returns, schedule)
    perturbed = an.simulate(perturbed_returns, rf_prev, schedule,
                            perturbed_strategies[name])

    before = base.index < cutoff
    deviation = np.abs(base.loc[before, "net_return"].to_numpy()
                       - perturbed.loc[before, "net_return"].to_numpy()).max()
    assert deviation < FLOAT_NOISE_FLOOR, (
        f"{name} leaks future information into returns before {cutoff.date()}: "
        f"max deviation {deviation:.2e}"
    )
    # Sanity: the perturbation does move later returns, by far more than the floor.
    assert np.abs(base["net_return"].to_numpy()
                  - perturbed["net_return"].to_numpy()).max() > 1e-4


def test_dynamic_selection_is_unchanged_by_future_returns():
    """The allocation chosen at each date must not depend on later returns."""
    returns, _ = an.load_inputs()
    schedule = an.build_schedule(returns.index)
    cutoff = pd.Timestamp("2018-01-02")

    perturbed_returns = returns.copy()
    perturbed_returns.loc[perturbed_returns.index >= cutoff] *= -3.0

    base_strategies, base_allocations = an.build_strategies(returns, schedule)
    perturbed_strategies, perturbed_allocations = an.build_strategies(
        perturbed_returns, schedule
    )
    # build_strategies records a decision only when its target function runs.
    for decision, _ in schedule:
        base_strategies["dynamic_es"](decision)
        perturbed_strategies["dynamic_es"](decision)

    base = {r["decision_date"]: r for r in base_allocations}
    other = {r["decision_date"]: r for r in perturbed_allocations}
    pre_cutoff = [d for d, _ in schedule if d < cutoff]
    assert len(pre_cutoff) > 0
    for decision in pre_cutoff:
        assert base[decision]["selected"] == other[decision]["selected"], decision
        # The Expected Shortfall inputs themselves must be bitwise identical.
        for name in an.CANDIDATES:
            assert base[decision][f"es_{name}"] == other[decision][f"es_{name}"], (
                f"{name} ES at {decision.date()} moved with future returns"
            )


# --- Metrics -----------------------------------------------------------------

def test_max_drawdown_on_a_hand_computable_path():
    # 1.0 -> 1.5 -> 0.75: peak 1.5, trough 0.75, drawdown -50%.
    returns = pd.Series([0.5, -0.5])
    assert an.max_drawdown(returns) == pytest.approx(-0.5)


def test_max_drawdown_is_zero_for_a_monotone_path():
    assert an.max_drawdown(pd.Series([0.01] * 10)) == pytest.approx(0.0)


def test_cagr_matches_compounded_growth():
    dates = _dates(an.TRADING_DAYS_PER_YEAR * 2)
    history = pd.DataFrame(
        {
            "net_return": 0.0,
            "traded_fraction": 0.0,
            "cost": 0.0,
            "gross_return": 0.0,
        },
        index=dates,
    )
    history["net_return"] = 0.001
    rf = _flat_rf(dates, 0.0)
    metrics = an.performance_metrics(history, rf)
    expected = (1.001 ** len(dates)) ** (1 / 2.0) - 1
    assert metrics["cagr"] == pytest.approx(expected)
    assert metrics["annualized_net_return"] == pytest.approx(
        0.001 * an.TRADING_DAYS_PER_YEAR
    )


def test_sharpe_is_zero_when_returns_match_the_risk_free_rate():
    dates = _dates(100)
    history = pd.DataFrame({"net_return": 0.0002, "traded_fraction": 0.0,
                            "cost": 0.0, "gross_return": 0.0002}, index=dates)
    rf = _flat_rf(dates, 0.0002)
    metrics = an.performance_metrics(history, rf)
    assert np.isnan(metrics["sharpe_ratio"]) or metrics["sharpe_ratio"] == pytest.approx(0.0)


def test_turnover_is_one_way(frozen):
    """Reported turnover is half the traded fraction, annualised (decision D6)."""
    _, rf_prev, _, histories, _ = frozen
    for name, history in histories.items():
        metrics = an.performance_metrics(history, rf_prev)
        years = len(history) / an.TRADING_DAYS_PER_YEAR
        expected = history["traded_fraction"].sum() / 2.0 / years
        assert metrics["annualized_turnover_one_way"] == pytest.approx(expected)


# --- Written results ---------------------------------------------------------

def test_result_files_exist_and_are_consistent():
    paths = {
        "metrics": RESULTS_DIR / "performance_metrics.csv",
        "daily": RESULTS_DIR / "daily_portfolio_returns.csv",
        "allocations": RESULTS_DIR / "dynamic_allocation_history.csv",
    }
    for label, path in paths.items():
        assert path.exists(), f"{label} missing at {path}; run 'python3 analysis.py'"

    metrics = pd.read_csv(paths["metrics"], index_col="strategy")
    daily = pd.read_csv(paths["daily"], index_col="Date", parse_dates=True)
    allocations = pd.read_csv(paths["allocations"], parse_dates=["decision_date",
                                                                "execution_date"])

    expected = ["traditional_60_40", "static_gold", "static_silver",
                "dynamic_es", "volatility_managed_60_40"]
    assert list(metrics.index) == expected
    assert list(daily.columns) == expected
    assert not daily.isna().any().any()
    assert (allocations["execution_date"] > allocations["decision_date"]).all()
    assert allocations["selected"].isin(an.CANDIDATES).all()

    # The written metrics must agree with the written daily series.
    for name in expected:
        assert metrics.loc[name, "n_days"] == len(daily)
        assert metrics.loc[name, "annualized_net_return"] == pytest.approx(
            daily[name].mean() * an.TRADING_DAYS_PER_YEAR
        )

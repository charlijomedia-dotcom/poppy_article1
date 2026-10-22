"""Phase 3 tests: bootstrap correctness, inference behaviour and output integrity.

The bootstrap is checked structurally (block geometry, reproducibility, shared
resamples) and statistically (centring, and that block resampling widens the
standard error of a dependent series relative to the i.i.d. formula). The
degenerate comparison — dynamic versus static gold, which are the same series —
is checked to yield no numerical p-value anywhere, including in the written
tables.

Synthetic series are test fixtures only and are never research data.

Run:  python3 -m pytest test_inference.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis as an
import inference as inf

RESULTS_DIR = inf.RESULTS_DIR


# --- Bootstrap structure -----------------------------------------------------

def test_index_block_has_the_right_shape_and_range():
    rng = np.random.default_rng(0)
    index = inf.bootstrap_index_block(50, 32, 1 / 21, rng)
    assert index.shape == (32, 50)
    assert index.min() >= 0 and index.max() < 50


def test_blocks_advance_by_one_and_wrap_circularly():
    """Within a block each step advances one position, modulo n."""
    rng = np.random.default_rng(1)
    n = 40
    index = inf.bootstrap_index_block(n, 64, 1 / 21, rng)
    step = (index[:, 1:] - index[:, :-1]) % n
    continued = step == 1
    # Steps are either a continuation (+1 mod n) or a fresh uniform draw.
    assert continued.mean() > 0.5, "most steps should continue their block"


def test_expected_block_length_matches_the_protocol():
    """Block lengths are geometric with mean 21 (decision D7)."""
    rng = np.random.default_rng(2)
    n = 2000
    index = inf.bootstrap_index_block(n, 400, 1 / inf.BLOCK_LENGTH, rng)
    step = (index[:, 1:] - index[:, :-1]) % n
    restart_rate = float((step != 1).mean())
    # A fresh draw can coincidentally equal the continuation value with
    # probability 1/n, so the measured rate is a slight underestimate.
    observed_block_length = 1.0 / restart_rate
    assert observed_block_length == pytest.approx(inf.BLOCK_LENGTH, rel=0.08)


def test_bootstrap_is_reproducible_under_the_fixed_seed():
    values = np.linspace(-0.01, 0.01, 300)
    first = inf.stationary_bootstrap_means({"x": values}, n_boot=200, seed=inf.SEED)
    second = inf.stationary_bootstrap_means({"x": values}, n_boot=200, seed=inf.SEED)
    assert np.array_equal(first["x"], second["x"])


def test_a_different_seed_gives_a_different_draw():
    values = np.linspace(-0.01, 0.01, 300)
    first = inf.stationary_bootstrap_means({"x": values}, n_boot=200, seed=inf.SEED)
    other = inf.stationary_bootstrap_means({"x": values}, n_boot=200, seed=inf.SEED + 1)
    assert not np.array_equal(first["x"], other["x"])


def test_all_series_share_the_same_resamples():
    """Comparisons must be drawn on identical replications."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y = rng.normal(size=400)
    alone = inf.stationary_bootstrap_means({"x": x}, n_boot=150, seed=inf.SEED)
    together = inf.stationary_bootstrap_means({"x": x, "y": y}, n_boot=150, seed=inf.SEED)
    assert np.array_equal(alone["x"], together["x"])


def test_bootstrap_rejects_series_of_unequal_length():
    with pytest.raises(ValueError):
        inf.stationary_bootstrap_means({"x": np.zeros(50), "y": np.zeros(60)},
                                       n_boot=10)


def test_bootstrap_of_a_constant_series_is_that_constant():
    values = np.full(200, 0.004)
    means = inf.stationary_bootstrap_means({"x": values}, n_boot=100, seed=inf.SEED)["x"]
    assert np.allclose(means, 0.004)


# --- Bootstrap statistical behaviour ----------------------------------------

def test_bootstrap_distribution_centres_on_the_sample_mean():
    rng = np.random.default_rng(4)
    values = rng.normal(0.001, 0.01, 1500)
    means = inf.stationary_bootstrap_means({"x": values}, n_boot=1000, seed=inf.SEED)["x"]
    assert means.mean() == pytest.approx(values.mean(), abs=0.0008)


def test_standard_error_matches_the_iid_formula_for_independent_data():
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 0.01, 3000)
    means = inf.stationary_bootstrap_means({"x": values}, n_boot=1500, seed=inf.SEED)["x"]
    iid_se = values.std(ddof=1) / np.sqrt(len(values))
    assert means.std(ddof=1) == pytest.approx(iid_se, rel=0.25)


def test_block_resampling_widens_the_standard_error_for_dependent_data():
    """The point of blocks: serial dependence inflates the true standard error."""
    rng = np.random.default_rng(6)
    n = 3000
    noise = rng.normal(0, 0.01, n)
    autocorrelated = np.empty(n)
    autocorrelated[0] = noise[0]
    for t in range(1, n):  # AR(1), rho = 0.9
        autocorrelated[t] = 0.9 * autocorrelated[t - 1] + noise[t]
    means = inf.stationary_bootstrap_means({"x": autocorrelated}, n_boot=1000,
                                           seed=inf.SEED)["x"]
    iid_se = autocorrelated.std(ddof=1) / np.sqrt(n)
    assert means.std(ddof=1) > 2.0 * iid_se


# --- Paired comparison -------------------------------------------------------

def test_identical_series_yield_no_p_value():
    """The dynamic/static-gold case: no sampling variation, so no test."""
    rng = np.random.default_rng(7)
    series = rng.normal(0, 0.01, 500)
    result = inf.paired_comparison(series, series.copy())
    assert result["status"] == "identical_series"
    assert np.isnan(result["p_value"])
    assert result["annualized_difference"] == 0.0
    assert result["n_days_differing"] == 0
    assert result["max_abs_daily_difference"] == 0.0


# --- Primary directional (one-sided) test ------------------------------------

def test_one_sided_p_value_is_hand_checkable():
    """Arithmetic worked out by hand on a crafted bootstrap distribution.

    The difference series is the constant 0.001, so the observed mean d is
    0.001. The five bootstrap means are recentred by subtracting d.

    One-sided (H1: > 0) counts recentred >= d, i.e. bootstrap mean >= 2d =
    0.002: that is 0.0025 and 0.0035, two of five, so
        p = (1 + 2) / (1 + 5) = 0.5.
    Two-sided counts |recentred| >= d, i.e. bootstrap mean <= 0 or >= 0.002:
    that is -0.0005, 0.0025 and 0.0035, three of five, so
        p = (1 + 3) / (1 + 5) = 2/3.

    Every value sits at least 0.0005 away from a decision boundary, so the
    counts are exact and cannot be flipped by floating-point representation.
    (The obvious choice of 0.002 as a bootstrap value would sit exactly on the
    one-sided threshold, where the mean of ten copies of 0.001 rounds to
    0.0010000000000000002 and tips the comparison.)
    """
    benchmark = np.zeros(10)
    dynamic = np.full(10, 0.001)
    boot = np.array([-0.0005, 0.0005, 0.0015, 0.0025, 0.0035])

    greater = inf.paired_comparison(dynamic, benchmark, boot_means=boot,
                                    alternative="greater")
    assert greater["alternative"] == "greater"
    assert greater["mean_daily_difference"] == pytest.approx(0.001)
    assert greater["p_value"] == pytest.approx(3 / 6)

    two_sided = inf.paired_comparison(dynamic, benchmark, boot_means=boot)
    assert two_sided["alternative"] == "two-sided"
    assert two_sided["p_value"] == pytest.approx(4 / 6)


def test_one_sided_interval_is_a_lower_bound():
    """A directional test reports a lower bound, not a two-sided interval."""
    benchmark = np.zeros(10)
    dynamic = np.full(10, 0.001)
    boot = np.array([-0.0005, 0.0005, 0.0015, 0.0025, 0.0035])
    greater = inf.paired_comparison(dynamic, benchmark, boot_means=boot,
                                    alternative="greater")
    assert np.isinf(greater["ci_high"])
    assert np.isfinite(greater["ci_low"])

    two_sided = inf.paired_comparison(dynamic, benchmark, boot_means=boot)
    assert np.isfinite(two_sided["ci_high"])


def test_unsupported_alternative_is_rejected():
    with pytest.raises(ValueError):
        inf.paired_comparison(np.zeros(5), np.ones(5), boot_means=np.zeros(4),
                              alternative="less")


def test_one_sided_test_detects_a_seeded_positive_edge():
    """A genuine positive edge: one-sided evidence is stronger than two-sided."""
    rng = np.random.default_rng(inf.SEED)
    # The two legs must be independent, so the *difference* has real variance.
    # Shifting one series by a constant would make the difference constant, and
    # a zero-variance difference pins both p-values to the resolution floor.
    benchmark = rng.normal(0, 0.005, 1200)
    # Calibrated so neither p-value hits the floor: the realised sample mean of
    # this seeded draw is about 0.00052 a day, t of roughly 2.6.
    dynamic = rng.normal(0, 0.005, 1200) + 0.0002
    greater = inf.paired_comparison(dynamic, benchmark, alternative="greater")
    two_sided = inf.paired_comparison(dynamic, benchmark)
    assert greater["p_value"] > greater["p_value_resolution"], "not at the floor"
    assert greater["p_value"] < two_sided["p_value"]
    assert greater["p_value"] < 0.05
    assert greater["ci_low"] > 0


def test_one_sided_test_does_not_reward_a_negative_effect():
    """With the point estimate below zero, superiority cannot be supported."""
    rng = np.random.default_rng(inf.SEED)
    benchmark = rng.normal(0, 0.005, 1200)
    worse = rng.normal(0, 0.005, 1200) - 0.0005
    greater = inf.paired_comparison(worse, benchmark, alternative="greater")
    assert greater["annualized_difference"] < 0
    assert greater["p_value"] > 0.5
    assert greater["ci_low"] < 0


def test_one_sided_is_about_half_the_two_sided_for_a_symmetric_draw():
    rng = np.random.default_rng(99)
    # Independent legs again, so the difference is not degenerate and the two
    # p-values are genuinely comparable rather than both pinned to the floor.
    benchmark = rng.normal(0, 0.01, 2000)
    dynamic = rng.normal(0, 0.01, 2000) + 0.0002
    boot = inf.stationary_bootstrap_means(
        {"d": (dynamic - benchmark)}, n_boot=4000, seed=inf.SEED
    )["d"]
    greater = inf.paired_comparison(dynamic, benchmark, boot_means=boot,
                                    alternative="greater")
    two_sided = inf.paired_comparison(dynamic, benchmark, boot_means=boot)
    assert greater["p_value"] > greater["p_value_resolution"]
    assert two_sided["p_value"] > two_sided["p_value_resolution"]
    assert greater["p_value"] == pytest.approx(two_sided["p_value"] / 2, abs=0.05)


def test_identical_series_give_no_p_value_under_the_directional_test():
    """The dynamic/static-gold treatment is preserved for the primary test too."""
    rng = np.random.default_rng(12)
    series = rng.normal(0, 0.01, 400)
    result = inf.paired_comparison(series, series.copy(), alternative="greater")
    assert result["status"] == "identical_series"
    assert np.isnan(result["p_value"])
    assert result["alternative"] == "greater"


def test_a_real_difference_is_detected():
    rng = np.random.default_rng(8)
    base = rng.normal(0, 0.005, 1200)
    shifted = base + 0.0015  # a large, persistent daily edge
    result = inf.paired_comparison(shifted, base, boot_means=None)
    assert result["status"] == "tested"
    assert result["annualized_difference"] == pytest.approx(
        0.0015 * an.TRADING_DAYS_PER_YEAR
    )
    assert result["ci_low"] > 0, "a constant positive edge should exclude zero"
    assert result["p_value"] <= 2 * result["p_value_resolution"]


def test_pure_noise_is_not_declared_different():
    rng = np.random.default_rng(9)
    a = rng.normal(0, 0.01, 2000)
    b = rng.normal(0, 0.01, 2000)
    result = inf.paired_comparison(a, b)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["p_value"] > 0.05


def test_p_value_is_bounded_and_never_zero():
    rng = np.random.default_rng(10)
    a = rng.normal(0, 0.01, 600)
    b = a + 0.01
    result = inf.paired_comparison(a, b)
    assert 0 < result["p_value"] <= 1.0
    assert result["p_value"] >= result["p_value_resolution"]


def test_confidence_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(11)
    a = rng.normal(0.0005, 0.01, 1500)
    b = rng.normal(0.0000, 0.01, 1500)
    result = inf.paired_comparison(a, b)
    assert result["ci_low"] <= result["annualized_difference"] <= result["ci_high"]


# --- Selection behaviour -----------------------------------------------------

@pytest.fixture(scope="module")
def frozen_runs():
    returns, rf_prev = an.load_inputs()
    schedule = an.build_schedule(returns.index)
    runs = {
        cost: inf.run_cost_level(returns, rf_prev, schedule, cost)
        for cost in inf.COST_LEVELS_BPS
    }
    return runs


def test_selection_frequency_accounts_for_every_decision(frozen_runs):
    _, _, _, allocations = frozen_runs[inf.PRINCIPAL_COST_BPS]
    frequency, advantage, summary = inf.selection_summary(allocations)
    assert frequency["decisions_selected"].sum() == len(allocations)
    assert frequency["share_of_decisions"].sum() == pytest.approx(1.0)
    assert len(advantage) == len(allocations)


def test_selected_candidate_always_has_the_least_severe_shortfall(frozen_runs):
    _, _, _, allocations = frozen_runs[inf.PRINCIPAL_COST_BPS]
    _, advantage, _ = inf.selection_summary(allocations)
    # Expected Shortfall is negative; the rule picks the largest (least severe).
    assert (advantage["es_selected"] >= advantage["es_runner_up"]).all()
    assert (advantage["es_margin"] > 0).all(), "every selection should be strict"


def test_selection_is_independent_of_the_cost_level(frozen_runs):
    """The rule ranks gross candidate returns, so costs cannot move it."""
    reference = [r["selected"] for r in frozen_runs[inf.COST_LEVELS_BPS[0]][3]]
    for cost in inf.COST_LEVELS_BPS[1:]:
        assert [r["selected"] for r in frozen_runs[cost][3]] == reference


def test_dynamic_matches_static_gold_at_every_cost_level(frozen_runs):
    """The documented finding, asserted rather than assumed."""
    for cost in inf.COST_LEVELS_BPS:
        _, _, daily, _ = frozen_runs[cost]
        assert np.array_equal(daily[inf.DYNAMIC].to_numpy(),
                              daily["static_gold"].to_numpy()), cost


# --- Cost robustness ---------------------------------------------------------

def test_higher_costs_strictly_reduce_net_returns(frozen_runs):
    for strategy in inf.STRATEGY_LABELS:
        returns_by_cost = [
            frozen_runs[cost][1].loc[strategy, "annualized_net_return"]
            for cost in inf.COST_LEVELS_BPS
        ]
        assert returns_by_cost == sorted(returns_by_cost, reverse=True), strategy


def test_cost_drag_scales_with_the_cost_level(frozen_runs):
    """Drag is proportional to the rate, since turnover does not depend on it."""
    for strategy in inf.STRATEGY_LABELS:
        drag_10 = frozen_runs[10][1].loc[strategy, "annualized_cost_drag"]
        drag_50 = frozen_runs[50][1].loc[strategy, "annualized_cost_drag"]
        assert drag_50 == pytest.approx(5.0 * drag_10, rel=1e-9), strategy


def test_gross_returns_do_not_depend_on_costs(frozen_runs):
    for strategy in inf.STRATEGY_LABELS:
        values = [frozen_runs[cost][1].loc[strategy, "annualized_gross_return"]
                  for cost in inf.COST_LEVELS_BPS]
        assert all(v == pytest.approx(values[0], rel=1e-12) for v in values), strategy


# --- Written outputs ---------------------------------------------------------

EXPECTED_OUTPUTS = [
    "table1_descriptive_statistics.csv", "table1_descriptive_statistics.tex",
    "table1b_correlations.csv", "table1b_correlations.tex",
    "table1c_sample_notes.csv",
    "table2_performance.csv", "table2_performance.tex",
    "table3a_paired_comparisons.csv", "table3a_paired_comparisons.tex",
    "table3b_cost_robustness.csv", "table3b_cost_robustness.tex",
    "bootstrap_inference.csv", "selection_frequency.csv",
    "es_advantage_history.csv", "es_advantage_summary.csv",
    "performance_all_cost_levels.csv",
    "figure1_cumulative_wealth.pdf", "figure1_cumulative_wealth.png",
    "figure2_es_differences.pdf", "figure2_es_differences.png",
]


@pytest.mark.parametrize("name", EXPECTED_OUTPUTS)
def test_expected_output_exists(name):
    path = RESULTS_DIR / name
    assert path.exists(), f"{path} missing; run 'python3 inference.py'"
    assert path.stat().st_size > 80, f"{path} looks truncated"


def test_inference_csv_records_every_comparison_at_every_cost_level():
    frame = pd.read_csv(RESULTS_DIR / "bootstrap_inference.csv")
    # Every benchmark at every cost level, plus the one primary directional row.
    assert len(frame) == len(inf.COST_LEVELS_BPS) * len(inf.BENCHMARKS) + 1
    assert set(frame["cost_bps"]) == set(inf.COST_LEVELS_BPS)
    identical = frame[frame["status"] == "identical_series"]
    assert len(identical) == len(inf.COST_LEVELS_BPS)
    assert (identical["benchmark"] == "static_gold").all()
    assert identical["p_value"].isna().all(), "no p-value for identical series"
    tested = frame[frame["status"] == "tested"]
    assert tested["p_value"].notna().all()
    assert ((tested["p_value"] > 0) & (tested["p_value"] <= 1)).all()


def test_published_table_shows_no_p_value_for_the_identical_comparison():
    table = pd.read_csv(RESULTS_DIR / "table3a_paired_comparisons.csv",
                        keep_default_na=False)
    identical = table[table["Status"] == "identical_series"]
    assert len(identical) == len(inf.COST_LEVELS_BPS)
    assert (identical["p-value"] == "n/a").all()
    assert (identical["95% bootstrap CI (pp)"] == "--").all()
    latex = (RESULTS_DIR / "table3a_paired_comparisons.tex").read_text()
    assert "n/a" in latex

    # A reader using pandas defaults gets NaN for those rows, never a number.
    default_read = pd.read_csv(RESULTS_DIR / "table3a_paired_comparisons.csv")
    assert default_read.loc[
        default_read["Status"] == "identical_series", "p-value"
    ].isna().all()


def test_exported_inference_marks_exactly_one_primary_test():
    frame = pd.read_csv(RESULTS_DIR / "bootstrap_inference.csv")
    primary = frame[frame["test_role"] == "primary"]
    assert len(primary) == 1
    row = primary.iloc[0]
    assert row["benchmark"] == inf.PRIMARY_BENCHMARK
    assert row["cost_bps"] == inf.PRIMARY_COST_BPS
    assert row["alternative"] == "greater"
    assert row["status"] == "tested"
    assert np.isinf(row["ci_high"])
    # Supplementary rows stay two-sided and keep their multiplicity adjustment.
    supplementary = frame[frame["test_role"] == "supplementary"]
    assert (supplementary["alternative"] == "two-sided").all()
    assert len(supplementary) == len(inf.COST_LEVELS_BPS) * len(inf.BENCHMARKS)
    assert np.isnan(row["bonferroni_p_value"]), (
        "a single prespecified test carries no multiplicity burden"
    )


def test_published_table_distinguishes_primary_from_supplementary():
    table = pd.read_csv(RESULTS_DIR / "table3a_paired_comparisons.csv",
                        keep_default_na=False)
    assert "Test" in table.columns and "Hypothesis" in table.columns
    primary = table[table["Test"] == "Primary"]
    assert len(primary) == 1
    assert primary.iloc[0]["Hypothesis"] == "one-sided (H1: > 0)"
    assert primary.iloc[0]["Comparison"] == "Dynamic ES - Traditional 60/40"
    assert "inf)" in primary.iloc[0]["95% bootstrap CI (pp)"]
    assert (table[table["Test"] == "Supplementary"]["Hypothesis"]
            == "two-sided (H1: != 0)").all()
    latex = (RESULTS_DIR / "table3a_paired_comparisons.tex").read_text()
    assert "PRIMARY" in latex and "SUPPLEMENTARY" in latex


def test_primary_and_supplementary_agree_on_the_point_estimate():
    """Same data and same resamples: only the hypothesis differs."""
    frame = pd.read_csv(RESULTS_DIR / "bootstrap_inference.csv")
    primary = frame[frame["test_role"] == "primary"].iloc[0]
    twin = frame[(frame["test_role"] == "supplementary")
                 & (frame["benchmark"] == inf.PRIMARY_BENCHMARK)
                 & (frame["cost_bps"] == inf.PRIMARY_COST_BPS)].iloc[0]
    assert primary["annualized_difference"] == pytest.approx(
        twin["annualized_difference"]
    )
    assert primary["n_days"] == twin["n_days"]


def test_performance_table_separates_arithmetic_mean_from_cagr():
    table = pd.read_csv(RESULTS_DIR / "table2_performance.csv", index_col="Strategy")
    assert "Ann. mean net return (%)" in table.columns
    assert "CAGR (%)" in table.columns
    # They are different quantities and must not be reported as interchangeable.
    assert (table["Ann. mean net return (%)"] != table["CAGR (%)"]).all()
    # Arithmetic mean exceeds CAGR whenever returns vary (Jensen's inequality).
    assert (table["Ann. mean net return (%)"] > table["CAGR (%)"]).all()


def test_latex_tables_are_well_formed():
    for name in [n for n in EXPECTED_OUTPUTS if n.endswith(".tex")]:
        text = (RESULTS_DIR / name).read_text()
        assert text.count(r"\begin{table}") == 1
        assert text.count(r"\end{table}") == 1
        assert r"\toprule" in text and r"\bottomrule" in text
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1


def test_descriptive_table_covers_all_four_instruments():
    table = pd.read_csv(RESULTS_DIR / "table1_descriptive_statistics.csv",
                        index_col="Instrument")
    assert list(table.index) == an.ASSETS
    assert (table["Obs."] == table["Obs."].iloc[0]).all()
    correlations = pd.read_csv(RESULTS_DIR / "table1b_correlations.csv",
                               index_col="Instrument")
    assert list(correlations.index) == an.ASSETS
    assert np.allclose(np.diag(correlations.to_numpy()), 1.0)

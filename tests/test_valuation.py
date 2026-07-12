import pytest

from utils.valuation import (
    calc_cv,
    calc_dcf,
    compute_roic_series,
    compute_year_metrics,
)


# ─── calc_cv ────────────────────────────────────────────────────────────────
def test_calc_cv_matches_gordon_growth_formula():
    noplat_next, g_pct, roic_cv_pct, wacc = 1000.0, 4.0, 15.0, 0.12
    cv = calc_cv(noplat_next, g_pct, roic_cv_pct, wacc)
    reinv = (g_pct / 100) / (roic_cv_pct / 100)
    expected = noplat_next * (1 - reinv) / (wacc - g_pct / 100)
    assert cv == pytest.approx(expected)


def test_calc_cv_none_when_wacc_at_or_below_g():
    assert calc_cv(1000.0, 6.0, 15.0, 0.06) is None
    assert calc_cv(1000.0, 6.0, 15.0, 0.05) is None


def test_calc_cv_reinvestment_capped_at_99_percent():
    # g >> roic_cv would imply reinvesting >100% of NOPLAT — capped at 99%.
    cv = calc_cv(1000.0, 6.0, 1.0, 0.10)
    expected = 1000.0 * 0.01 / (0.10 - 0.06)
    assert cv == pytest.approx(expected)


def test_calc_cv_zero_roic_cv_means_no_reinvestment():
    cv = calc_cv(1000.0, 4.0, 0.0, 0.10)
    assert cv == pytest.approx(1000.0 / (0.10 - 0.04))


# ─── calc_dcf ───────────────────────────────────────────────────────────────
def test_calc_dcf_ev_equals_pv_explicit_plus_pv_cv():
    pv_exp, pv_cv, ev, rows, cv, noplat_11 = calc_dcf(
        noplat0=1000, g1_pct=10, g2_pct=5, gt_pct=3.5, wacc_dec=0.15, roic_cv_pct=15
    )
    assert ev == pytest.approx(pv_exp + pv_cv)
    assert len(rows) == 10


def test_calc_dcf_uses_g1_through_year_5_and_g2_after():
    _, _, _, rows, _, _ = calc_dcf(
        noplat0=1000, g1_pct=10, g2_pct=4, gt_pct=3.0, wacc_dec=0.14, roic_cv_pct=14
    )
    assert [r["g_pct"] for r in rows[:5]] == [10.0] * 5
    assert [r["g_pct"] for r in rows[5:]] == [4.0] * 5


def test_calc_dcf_noplat_11_compounds_both_growth_phases_plus_terminal():
    noplat0, g1, g2, gt = 1000.0, 10.0, 4.0, 3.0
    _, _, _, _, _, noplat_11 = calc_dcf(
        noplat0, g1, g2, gt, wacc_dec=0.14, roic_cv_pct=14
    )
    expected = noplat0 * (1 + g1 / 100) ** 5 * (1 + g2 / 100) ** 5 * (1 + gt / 100)
    assert noplat_11 == pytest.approx(expected)


def test_calc_dcf_reinvestment_rate_capped_at_95_percent_in_explicit_period():
    # Very high growth vs low ROIC would demand >100% reinvestment; capped at 95%,
    # so FCF is never less than 5% of NOPLAT for a given projected year.
    _, _, _, rows, _, _ = calc_dcf(
        noplat0=1000, g1_pct=20, g2_pct=20, gt_pct=3.0, wacc_dec=0.14, roic_cv_pct=14,
        roic_proj_pct=1.0,
    )
    for r in rows:
        assert r["reinv_pct"] <= 95.0
        assert r["fcf"] == pytest.approx(r["noplat"] * 0.05)


def test_calc_dcf_roic_proj_defaults_to_roic_cv_when_not_given():
    with_default = calc_dcf(1000, 10, 5, 3.5, 0.15, roic_cv_pct=12)
    explicit = calc_dcf(1000, 10, 5, 3.5, 0.15, roic_cv_pct=12, roic_proj_pct=12)
    assert with_default[2] == pytest.approx(explicit[2])


def test_calc_dcf_higher_roic_cv_increases_terminal_value():
    low_roic = calc_dcf(1000, 10, 5, 3.5, 0.15, roic_cv_pct=6)
    high_roic = calc_dcf(1000, 10, 5, 3.5, 0.15, roic_cv_pct=20)
    assert high_roic[1] > low_roic[1]  # pv_cv


# ─── compute_year_metrics ────────────────────────────────────────────────────
def _base_year_kwargs(**overrides):
    kwargs = dict(
        revenue=1_000_000.0,
        ebit=200_000.0,
        pretax_income=180_000.0,
        tax_expense=54_000.0,  # 30% effective rate
        da=50_000.0,
        capex=-80_000.0,
        delta_wc=-10_000.0,
        ppe=500_000.0,
        goodwill=100_000.0,
        current_assets=300_000.0,
        current_liabilities=150_000.0,
        cash=200_000.0,
        debt_short=20_000.0,
        debt_long=180_000.0,
        equity=600_000.0,
        interest_expense=15_000.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_compute_year_metrics_default_tax_rate_when_missing():
    m = compute_year_metrics(**_base_year_kwargs(pretax_income=None, tax_expense=None))
    assert m["tax_rate"] == 0.34


def test_compute_year_metrics_tax_rate_clamped_to_10_40_range():
    m_low = compute_year_metrics(
        **_base_year_kwargs(pretax_income=1_000_000.0, tax_expense=1_000.0)
    )
    assert m_low["tax_rate"] == 0.10

    m_high = compute_year_metrics(
        **_base_year_kwargs(pretax_income=100_000.0, tax_expense=90_000.0)
    )
    assert m_high["tax_rate"] == 0.40


def test_compute_year_metrics_noplat_uses_effective_tax_rate():
    m = compute_year_metrics(**_base_year_kwargs())
    assert m["tax_rate"] == pytest.approx(0.30)
    assert m["noplat"] == pytest.approx(200_000.0 * (1 - 0.30))


def test_compute_year_metrics_fcf_adds_delta_wc_not_subtracts():
    # delta_wc already carries the DFC-indireto cash-flow sign; a negative value
    # (working capital consuming cash) must reduce FCF when added, matching the
    # historical bug this module fixes.
    positive_wc = compute_year_metrics(**_base_year_kwargs(delta_wc=10_000.0))
    negative_wc = compute_year_metrics(**_base_year_kwargs(delta_wc=-10_000.0))
    assert positive_wc["fcf"] - negative_wc["fcf"] == pytest.approx(20_000.0)


def test_compute_year_metrics_capex_uses_absolute_value():
    m = compute_year_metrics(**_base_year_kwargs(capex=-80_000.0))
    assert m["capex"] == 80_000.0


def test_compute_year_metrics_wco_excludes_only_excess_cash():
    # cash_op floor = 1.5% of revenue = 15,000. Cash of 200,000 leaves 185,000
    # "excess" cash excluded from WCO.
    m = compute_year_metrics(**_base_year_kwargs())
    expected_wco = 300_000.0 - 150_000.0 - (200_000.0 - 15_000.0)
    assert m["wco"] == pytest.approx(expected_wco)


def test_compute_year_metrics_wco_keeps_cash_below_operating_floor():
    # When cash is under the operating floor, none of it is excluded (max(...,0)).
    m = compute_year_metrics(**_base_year_kwargs(cash=5_000.0))
    expected_wco = 300_000.0 - 150_000.0 - 0
    assert m["wco"] == pytest.approx(expected_wco)


def test_compute_year_metrics_ic_sums_wco_ppe_goodwill():
    m = compute_year_metrics(**_base_year_kwargs())
    assert m["ic"] == pytest.approx(m["wco"] + 500_000.0 + 100_000.0)


def test_compute_year_metrics_debt_sums_short_and_long():
    m = compute_year_metrics(**_base_year_kwargs())
    assert m["debt"] == pytest.approx(200_000.0)


# ─── compute_roic_series ─────────────────────────────────────────────────────
def test_compute_roic_series_first_year_has_no_roic():
    years = [
        {"noplat": 100.0, "ic": 1_000_000.0, "goodwill": 100_000.0},
        {"noplat": 120.0, "ic": 1_100_000.0, "goodwill": 100_000.0},
    ]
    result = compute_roic_series(years)
    assert result[0]["roic"] is None
    assert result[0]["roic_no_gw"] is None


def test_compute_roic_series_uses_prior_year_ic_as_denominator():
    years = [
        {"noplat": 100.0, "ic": 1_000_000.0, "goodwill": 200_000.0},
        {"noplat": 150.0, "ic": 1_100_000.0, "goodwill": 200_000.0},
    ]
    result = compute_roic_series(years)
    assert result[1]["roic"] == pytest.approx(150.0 / 1_000_000.0 * 100)
    assert result[1]["roic_no_gw"] == pytest.approx(
        150.0 / (1_000_000.0 - 200_000.0) * 100
    )


def test_compute_roic_series_none_when_prior_ic_below_threshold():
    years = [
        {"noplat": 100.0, "ic": 500.0, "goodwill": 0.0},  # below 1e4 threshold
        {"noplat": 150.0, "ic": 1_100_000.0, "goodwill": 0.0},
    ]
    result = compute_roic_series(years)
    assert result[1]["roic"] is None

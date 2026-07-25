"""Verify the numerical claims in NBEATS-Explorations/.../neurips_2026.tex against
the canonical CSVs under experiments/results/. Applies the strict health filter
(diverged | smape>100 | (best_epoch==0 & smape>50) | NaN) before aggregation.

Run from repo root:
    .venv/bin/python experiments/analysis/scripts/verify_paper_numbers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
M4 = ROOT / "experiments" / "results" / "m4"
WEATHER = ROOT / "experiments" / "results" / "weather"
TOURISM = ROOT / "experiments" / "results" / "tourism"
MILK = ROOT / "experiments" / "results" / "milk"
CONV_V2 = ROOT / "experiments" / "results" / "convergence_study_v2_results.csv"

TOL_SMAPE = 0.005
TOL_STD = 0.005
TOL_PARAMS_M = 0.05


def health_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "diverged" not in df.columns:
        return df
    m = (df["diverged"] != True) & df["smape"].notna() & (df["smape"] <= 100)
    if "best_epoch" in df.columns:
        m &= ~((df["best_epoch"] == 0) & (df["smape"] > 50))
    return df[m].copy()


def cell(csv: Path, config: str, period: str) -> tuple[float, float, int, float] | None:
    df = pd.read_csv(csv)
    sub = df[(df["config_name"] == config) & (df["period"] == period)]
    sub = health_filter(sub)
    if len(sub) == 0:
        return None
    params_m = float(sub["n_params"].iloc[0]) / 1e6
    return float(sub["smape"].mean()), float(sub["smape"].std()), len(sub), params_m


def cmp(label: str, paper: tuple, actual: tuple | None) -> bool:
    if actual is None:
        print(f"  ✗ {label}: not found in CSV")
        return False
    p_smape, p_std, p_n, p_params = paper
    a_smape, a_std, a_n, a_params = actual
    ok = (
        abs(a_smape - p_smape) < TOL_SMAPE
        and abs(a_std - p_std) < TOL_STD
        and a_n == p_n
        and abs(a_params - p_params) < TOL_PARAMS_M
    )
    mark = "✓" if ok else "✗"
    print(
        f"  {mark} {label}: paper {p_smape:.3f}±{p_std:.3f} ({p_n}, {p_params:.2f}M) "
        f"vs CSV {a_smape:.3f}±{a_std:.3f} ({a_n}, {a_params:.2f}M)"
    )
    return ok


def main() -> int:
    failures = 0

    print("Table 1 (main_leaderboard) — paper-sample winners")
    rows = [
        ("Yearly",    M4 / "comprehensive_m4_paper_sample_plateau_results.csv", "T+Coif2V3_30s_bdeq",     (13.542, 0.148, 10, 15.24)),
        ("Quarterly", M4 / "comprehensive_m4_paper_sample_plateau_results.csv", "NBEATS-IG_10s_ag0",      (10.312, 0.055, 10, 19.64)),
        ("Monthly",   M4 / "comprehensive_m4_paper_sample_plateau_results.csv", "TW_30s_td3_bdeq_sym10",  (13.240, 0.334,  9,  6.78)),
        ("Weekly",    M4 / "comprehensive_m4_paper_sample_results.csv",         "T+Coif2V3_30s_bdeq",     ( 6.735, 0.203, 10, 15.75)),
        ("Daily",     M4 / "tiered_offset_m4_allperiods_results.csv",           "T+Sym10V3_10s_tiered_ag0", ( 3.014, 0.035, 10,  5.25)),
        ("Hourly",    M4 / "comprehensive_m4_paper_sample_results.csv",         "NBEATS-IG_30s_agf",      ( 8.758, 0.099, 10, 43.58)),
    ]
    for period, csv, config, paper in rows:
        actual = cell(csv, config, period)
        if not cmp(f"{period} {config}", paper, actual):
            failures += 1

    print("\nTable 1 row Weather-96 (MSE)")
    df = pd.read_csv(WEATHER / "comprehensive_sweep_weather_results.csv")
    sub = df[df["config_name"] == "TAE+DB3V3AE_30s_ld8_ag0"]
    sub = health_filter(sub)
    if len(sub):
        mse_mean, mse_std = sub["mse"].mean(), sub["mse"].std()
        params_m = sub["n_params"].iloc[0] / 1e6
        ok = abs(mse_mean - 0.138) < 0.005 and abs(mse_std - 0.027) < 0.005 and len(sub) == 10 and abs(params_m - 7.1) < 0.1
        print(
            f"  {'✓' if ok else '✗'} Weather-96 TAE+DB3V3AE_30s_ld8_ag0: paper MSE 0.138±0.027 (10, 7.1M) "
            f"vs CSV {mse_mean:.3f}±{mse_std:.3f} ({len(sub)}, {params_m:.2f}M)"
        )
        if not ok:
            failures += 1
    else:
        print("  ✗ Weather-96 row: config not found")
        failures += 1

    print("\nTable 2 (sub-1M frontier)")
    sub1m = [
        ("Yearly",    M4 / "comprehensive_m4_paper_sample_plateau_results.csv", "TWAE_10s_ld32_ag0",         (13.546, 0.102, 10, 0.48)),
        ("Quarterly", M4 / "comprehensive_m4_paper_sample_results.csv",         "TWAE_10s_ld32_ag0",         (10.404, 0.028, 10, 0.49)),
        ("Monthly",   M4 / "comprehensive_m4_paper_sample_sym10_fills_results.csv", "TWAE_10s_ld32_sym10_ag0", (13.513, 0.378, 10, 0.58)),
        ("Weekly",    M4 / "comprehensive_m4_paper_sample_results.csv",         "TWAELG_10s_ld32_db3_ag0",   ( 7.252, 0.263, 10, 0.54)),
        ("Daily",     M4 / "comprehensive_m4_paper_sample_results.csv",         "TWGAELG_10s_ld16_db3_ag0",  ( 3.051, 0.049, 10, 0.52)),
        ("Hourly",    M4 / "comprehensive_m4_paper_sample_results.csv",         "TWAELG_10s_ld32_db3_agf",   ( 8.924, 0.129, 10, 0.85)),
    ]
    for period, csv, config, paper in sub1m:
        if not cmp(f"{period} {config}", paper, cell(csv, config, period)):
            failures += 1

    print("\nTable 6 (M4-Yearly top-5, paper-sample)")
    yearly = [
        (M4 / "comprehensive_m4_paper_sample_plateau_results.csv",    "T+Coif2V3_30s_bdeq",       (13.542, 0.148, 10, 15.24)),
        (M4 / "comprehensive_m4_paper_sample_plateau_results.csv",    "TWAE_10s_ld32_ag0",        (13.546, 0.102, 10,  0.48)),
        (M4 / "comprehensive_m4_paper_sample_results.csv",            "TALG+DB3V3ALG_10s_ag0",    (13.550, 0.096, 10,  1.04)),
        (M4 / "comprehensive_m4_paper_sample_plateau_results.csv",    "TWAE_10s_ld32_sym10_ag0",  (13.553, 0.165, 10,  0.48)),
        (M4 / "tiered_offset_m4_allperiods_paperlr_results.csv",      "T+DB3V3_10s_tiered_agf",   (13.554, 0.101, 10,  5.07)),
    ]
    for csv, config, paper in yearly:
        if not cmp(f"Yearly {config}", paper, cell(csv, config, "Yearly")):
            failures += 1

    print("\nTable 7 (M4-Quarterly top-5, paper-sample)")
    quarterly = [
        (M4 / "comprehensive_m4_paper_sample_plateau_results.csv",  "NBEATS-IG_10s_ag0",         (10.312, 0.055, 10, 19.64)),
        (M4 / "tiered_offset_m4_allperiods_results.csv",            "T+Sym10V3_10s_tiered_ag0",  (10.325, 0.053, 10,  5.11)),
        (M4 / "tiered_offset_m4_allperiods_paperlr_results.csv",    "T+DB3V3_30s_tiered_ag0",    (10.345, 0.065, 10, 15.34)),
        (M4 / "comprehensive_m4_paper_sample_plateau_results.csv",  "NBEATS-IG_10s_agf",         (10.354, 0.115, 10, 19.64)),
        (M4 / "tiered_offset_m4_allperiods_results.csv",            "T+DB3V3_10s_tiered_ag0",    (10.356, 0.053, 10,  5.11)),
    ]
    for csv, config, paper in quarterly:
        if not cmp(f"Quarterly {config}", paper, cell(csv, config, "Quarterly")):
            failures += 1

    print("\nTable 10 (M4-Daily top-5, paper-sample)")
    daily = [
        (M4 / "tiered_offset_m4_allperiods_results.csv", "T+Sym10V3_10s_tiered_ag0",     ( 3.014, 0.035, 10,  5.25)),
        (M4 / "tiered_offset_m4_allperiods_results.csv", "T+DB3V3_10s_tiered_ag0",       ( 3.023, 0.039, 10,  5.25)),
        (M4 / "tiered_offset_m4_allperiods_results.csv", "T+Sym10V3_30s_tiered_ag0",     ( 3.031, 0.040, 10, 15.75)),
        (M4 / "tiered_offset_m4_allperiods_results.csv", "TAELG+Sym10V3AELG_30s_tiered", ( 3.035, 0.033, 10,  3.41)),
        (M4 / "comprehensive_m4_paper_sample_results.csv", "TAELG+Coif2V3ALG_30s_ag0",   ( 3.036, 0.034, 10,  3.73)),
    ]
    for csv, config, paper in daily:
        if not cmp(f"Daily {config}", paper, cell(csv, config, "Daily")):
            failures += 1

    print("\nTable 14 (active_g M4-Yearly with sMAPE<20 outlier filter)")
    df = pd.read_csv(CONV_V2)
    yearly = df[df["period"] == "Yearly"]
    for ag, label, claim in [("False", "baseline", (13.616, 191)), ("forecast", "agf", (13.669, 200))]:
        s = yearly[yearly["active_g"] == ag]
        s = s[s["smape"] < 20]
        actual_mean = s["smape"].mean()
        ok = abs(actual_mean - claim[0]) < 0.005 and len(s) == claim[1]
        print(f"  {'✓' if ok else '✗'} M4-Yearly {label} (active_g={ag}): paper {claim[0]:.3f} (n={claim[1]}) vs CSV {actual_mean:.3f} (n={len(s)})")
        if not ok:
            failures += 1

    print("\nTable 17 (Tourism-Yearly)")
    df = pd.read_csv(TOURISM / "comprehensive_sweep_tourism_results.csv")
    sub = df[(df["config_name"] == "TW_10s_td3_bdeq_db3") & (df["period"] == "Tourism-Yearly")]
    sub = health_filter(sub)
    if len(sub):
        mean, std, n, params_m = sub["smape"].mean(), sub["smape"].std(), len(sub), sub["n_params"].iloc[0] / 1e6
        ok = abs(mean - 21.773) < 0.005 and abs(std - 0.384) < 0.005 and n == 10 and abs(params_m - 2.0) < 0.1
        print(f"  {'✓' if ok else '✗'} Tourism-Yearly TW_10s_td3_bdeq_db3: paper 21.773±0.384 (10, 2.0M) vs CSV {mean:.3f}±{std:.3f} ({n}, {params_m:.2f}M)")
        if not ok:
            failures += 1
    else:
        print("  ✗ Tourism-Yearly TW_10s_td3_bdeq_db3: not found")
        failures += 1

    print("\nTable 18 (Milk 10-stack)")
    df = pd.read_csv(MILK / "comprehensive_sweep_milk_results.csv")
    sub = df[df["config_name"] == "TALG+DB3V3ALG_10s_ag0"]
    sub = health_filter(sub)
    if len(sub):
        mean, std, n, params_m = sub["smape"].mean(), sub["smape"].std(), len(sub), sub["n_params"].iloc[0] / 1e6
        ok = abs(mean - 1.512) < 0.005 and abs(std - 0.572) < 0.005 and n == 10 and abs(params_m - 1.0) < 0.1
        print(f"  {'✓' if ok else '✗'} Milk TALG+DB3V3ALG_10s_ag0: paper 1.512±0.572 (10, 1.0M) vs CSV {mean:.3f}±{std:.3f} ({n}, {params_m:.2f}M)")
        if not ok:
            failures += 1

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} mismatch(es)")
    return failures


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)

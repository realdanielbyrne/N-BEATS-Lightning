# Submission Checklist — N-BEATS Architecture Exploration Paper

> **Status snapshot (Feb 2025):** Paper ~85% written, ~18% of experiments complete.
> Track progress by checking boxes as items are finished.

---

## 1. Statistical Power — Increase Seeds (3 → 5)

All experiment parts currently use seeds `[42, 43, 44]`. Adding seeds `45` and `46` only requires changing `N_RUNS` and rerunning — the script's `result_exists()` check automatically skips completed runs and `append_result()` adds only the new seed rows to existing CSVs.

- [ ] Update `N_RUNS` constant in `run_experiments.py` (3 → 5)
- [ ] Rerun Part 1 for completed periods (Yearly, Quarterly, Monthly) — only seeds 45/46 will execute
- [ ] For Part 3 (Ensemble): delete existing summary rows from `ensemble_summary_results.csv` for any already-aggregated periods (Yearly) so the ensemble OWA is recomputed over all 5 seeds
- [ ] Part 6 (Convergence Study) already uses 10 seeds — no change needed
- [ ] Update Table 4 (Training Protocol) in paper: "Seeds: 5 runs (42–46)"

---

## 2. Complete Existing Experiments

### Part 1: Block Benchmark (17 configurations × 6 periods × 5 seeds)

| Period    | Status       |
|-----------|-------------|
| Yearly    | ✅ Done (3 seeds — needs 2 more) |
| Quarterly | ✅ Done (3 seeds — needs 2 more) |
| Monthly   | ✅ Done (3 seeds — needs 2 more) |
| Weekly    | ❌ Not started |
| Daily     | ❌ Not started |
| Hourly    | ❌ Not started |

- [ ] Run Part 1: Weekly
- [ ] Run Part 1: Daily
- [ ] Run Part 1: Hourly

### Part 2: Ablation Studies (8 configurations × 6 periods × 5 seeds)

Tests `active_g`, `sum_losses`, and alternative activations (GELU, ELU, LeakyReLU, SELU).

- [ ] Run Part 2: Yearly
- [ ] Run Part 2: Quarterly
- [ ] Run Part 2: Monthly
- [ ] Run Part 2: Weekly
- [ ] Run Part 2: Daily
- [ ] Run Part 2: Hourly

### Part 3: Multi-Horizon Ensemble (3 architectures × 6 multipliers × 6 periods × 5 seeds)

| Period    | Status       |
|-----------|-------------|
| Yearly    | ✅ Done (3 seeds — needs 2 more) |
| Quarterly | ⚠️ Partial |
| Monthly   | ❌ Not started |
| Weekly    | ❌ Not started |
| Daily     | ❌ Not started |
| Hourly    | ❌ Not started |

- [ ] Complete Part 3: Quarterly
- [ ] Run Part 3: Monthly
- [ ] Run Part 3: Weekly
- [ ] Run Part 3: Daily
- [ ] Run Part 3: Hourly

### Part 4: Wavelet V2 Benchmark (8 configurations × 6 periods × 5 seeds)

Numerically stabilized wavelets with spectral normalization, LayerNorm, Xavier init, output clamping.

- [ ] Run Part 4: all periods

### Part 5: Wavelet V3 Benchmark (15 configurations × 6 periods × 5 seeds)

Orthonormal DWT basis with impulse-response synthesis + SVD orthogonalization.

- [ ] Run Part 5: all periods

### Part 6: Convergence Study (4 configurations × 2 datasets × 50 seeds)

2×2 factorial: `active_g` × `sum_losses` on 10-stack Generic. Multi-dataset (M4-Yearly + Weather-96), 50 random seeds per config.

- [x] Run Part 6: M4-Yearly (200 runs complete)
- [x] Run Part 6: Weather-96 (200 runs complete — replaced original Quarterly with Weather for cross-domain generalizability)

---

## 3. Additional Datasets (Beyond M4)

Strengthens generalizability claims. Consider one or two from:

- [ ] Decide which additional datasets to include
- [ ] ETTh1/ETTh2 (Electricity Transformer Temperature — standard long-horizon benchmark)
- [ ] Weather (21 meteorological variables, common in Transformer papers)
- [ ] Traffic (862 sensors, high-dimensional)
- [ ] ECL / Electricity (321 clients)
- [ ] Write new DataModule or adapt existing loaders
- [ ] Run baseline configs on chosen dataset(s)
- [ ] Add results section/table to paper

---

## 4. Competitor / External Comparisons

Current paper compares block types within N-BEATS only. Reviewers will expect context against modern alternatives.

- [ ] Decide comparison scope (literature-reported numbers vs. reproduced runs)
- [ ] N-HiTS — hierarchical interpolation (closest relative to N-BEATS)
- [ ] PatchTST — Transformer-based patch approach
- [ ] DLinear — simple linear baseline (often surprisingly strong)
- [ ] TimesNet / iTransformer (optional, if targeting top venue)
- [ ] Add comparison table to paper (Table in Section 5 or new Section 5.x)
- [ ] Discuss in Related Work and Conclusions

---

## 5. Figures

All 9 figures are currently marked "[To be produced]" in the paper.

- [ ] **Figure 1:** N-BEATS Architecture Diagram — block diagram of doubly residual topology with insets for Generic/BottleneckGeneric, Trend/Seasonality, Wavelet basis expansions
- [ ] **Figure 2:** OWA Comparison Bar Chart (Yearly) — configs sorted by mean OWA, ±1 std error bars, horizontal line at OWA=1.0
- [ ] **Figure 3:** Parameter Efficiency Scatter Plot — params (log scale) vs mean OWA, Pareto frontier highlighted
- [ ] **Figure 4:** OWA Heatmap (configs × periods) — green/red color scale, gray for diverged/NaN
- [ ] **Figure 5:** Training Stability Bar Chart — OWA std dev across seeds, highlight NBEATS-I-AE low variance
- [ ] **Figure 6:** Ensemble OWA vs Backcast Multiplier — 3 lines (G, I, I+G) with individual seed points (requires Part 3 completion)
- [ ] **Figure 7:** Ablation Grouped Bar Chart — per-period grouped bars for each ablation config (requires Part 2 completion)
- [ ] **Figure 8:** Bootstrap CI Forest Plot — 3 panels (Y/Q/M), rows = configs, 95% CIs (requires 5 seeds)
- [ ] **Figure 9:** OWA Distribution Box Plots — per-config faceted by period, jittered seed dots

### Figure production notes
- Export as vector graphics (PDF/SVG) for submission
- Consider a dedicated `figures/` script or Jupyter notebook for reproducibility

---

## 6. Paper Writing

### Tables to populate

- [ ] **Table 2 (Results):** Fill W/D/H columns once Weekly/Daily/Hourly runs complete
- [ ] **Table 6 (Ablation):** All cells currently `[pending]` — fill after Part 2 runs
- [ ] **Table 5 (Ensemble):** Add "all configs" ensemble row and remaining periods
- [ ] Update Table 3a/3b/3c with 5-seed statistics
- [ ] Add Table 3d/3e/3f for Weekly/Daily/Hourly detailed metrics

### Sections to finalize

- [ ] **Abstract** — currently `[TODO]`; write after all benchmarks complete
- [ ] **Section 5.3 (Ablation Studies)** — currently placeholder; write analysis after Part 2
- [ ] **Section 5.2.2 (Ensemble)** — expand with all-period results
- [ ] **Section 5.4 (Statistical Analysis)** — update with 5-seed bootstrap CIs and revised test results
- [ ] **Section 5.1.1** — resolve `[partial]` Trend+DB3Wavelet Monthly data
- [ ] **Section 6 (Conclusions)** — finalize after all results are in
- [ ] Remove all `[TODO]`, `[pending]`, and `[To be produced]` markers
- [ ] Proofread entire manuscript

### Optional enhancements

- [ ] Add parameter efficiency ratio (OWA / params) discussion
- [x] Add convergence rate analysis (Part 6 results) — Section 5.6 added to paper.md
- [ ] Add per-series distribution analysis (if implementing code changes)

---

## 7. Workshop Short Paper (4-page version)

Target: NeurIPS or ICML workshop on time series / efficient ML.

- [ ] Identify target workshop and submission deadline
- [ ] Distill to 4 pages: focus on novel blocks + key finding (block type doesn't matter much)
- [ ] Select 2–3 most compelling figures
- [ ] Condense tables to Yearly + one other period
- [ ] Trim related work to 1 paragraph
- [ ] Write workshop-specific abstract (150 words)
- [ ] Format per workshop template (typically NeurIPS or ICML style)
- [ ] Internal review pass
- [ ] Submit

---

## 8. Journal / Conference Full Paper

- [ ] Choose target venue (e.g., TMLR, Pattern Recognition, Neural Networks, AAAI/IJCAI)
- [ ] Format paper per venue style guide (LaTeX template)
- [ ] Convert `paper.md` to LaTeX if not already done
- [ ] Prepare supplementary materials:
  - [ ] Full per-period detailed tables
  - [ ] Wavelet failure mode analysis (extended)
  - [ ] Hyperparameter sensitivity (if space-limited in main paper)
  - [ ] Code repository link and reproducibility statement
- [ ] Write reproducibility checklist (required by most ML venues)
- [ ] Ensure all experimental claims have statistical backing (CIs, effect sizes)
- [ ] Internal review pass
- [ ] Submit

---

## Quick Reference — Experiment Commands

```bash
# Part 1: Block benchmark
python experiments/run_experiments.py --part 1 --periods Yearly Quarterly Monthly Weekly Daily Hourly

# Part 2: Ablation
python experiments/run_experiments.py --part 2 --periods Yearly Quarterly Monthly Weekly Daily Hourly

# Part 3: Multi-horizon ensemble
python experiments/run_experiments.py --part 3 --periods Yearly Quarterly Monthly Weekly Daily Hourly

# Part 4: Wavelet V2
python experiments/run_experiments.py --part 4 --periods Yearly Quarterly Monthly Weekly Daily Hourly

# Part 5: Wavelet V3
python experiments/run_experiments.py --part 5 --periods Yearly Quarterly Monthly Weekly Daily Hourly

# Part 6: Convergence study
python experiments/run_experiments.py --part 6 --periods Yearly Quarterly

# Run everything
python experiments/run_experiments.py --part all
```

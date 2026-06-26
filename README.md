# Protecting Customs Revenue with Scarce Audit Capacity: Selection-Robust Targeting of Post-Clearance Undervaluation under Label Bias

Reproducibility package for the paper *Protecting Customs Revenue with Scarce Audit Capacity: Selection-Robust Targeting of Post-Clearance Undervaluation under Label Bias* (sole author: Sushanta Paul).

The paper recasts customs audit targeting as estimation and decision-making under selective labels. Clearance-stage inspection is selective, confounded, and structurally blind to a band of declarations; post-clearance audit reaches the full population on recorded features with selection probabilities the administration sets. Using the post-clearance arm as a full-support, ignorably-assigned arm, the method defines an identified recoverable-duty estimand (cross-fitted doubly-robust), certifies it with sensitivity and assumption-free bounds, targets robustly, and attaches selection-valid conformal intervals with calibrated abstention. It also derives a minimum audit-allocation rule.

All experiments are simulation-based; no administrative data are used or required.

## Contents

| File | Reproduces |
|------|------------|
| `paper1_backbone.py` | Operating points (Section 5.1), identification ladder and estimator bias (Section 5.2, Table 2), held-out policy uplift (Section 5.3), and inspected-only evaluation bias (Section 5.4). |
| `paper1_policy.py` | Fifty-seed policy-value numbers reported in Section 5.3: realistic and benign top-five-percent capture and relative uplift (by mean of ratios and by ratio of means, with confidence intervals), the monotone-decreasing budget sweep over the top one to twenty percent, and the inspected-only evaluation bias. Defaults to fifty seeds; pass an integer first argument to override. |
| `paper1_prop6_verify.py` | Numerical confirmation of the minimum-audit-allocation rule, Proposition 6 (Section 4.5): variance scaling, the conformal informativeness floor, and that the derived rate predicts the operating boundary. |
| `paper1_adverse_grid.py` | Adverse-DGP band-intensity sweep, Table 7 (Section 6.5): the policy uplift as a monotone function of the recoverable duty concentrated in the blind band. |
| `paper1_consistency.py` | Finite-sample consistency check, Table 4 (Section 5.2): percent bias, standard error scaled by the square root of the sample size, and ninety-five-percent confidence-interval coverage of the design estimator across sample sizes, confirming the root-n rate of Proposition 1. |
| `paper1_bandsize.py` | Blind-band-size robustness sweep (Section 6.6): band share, the recoverable duty the band conceals, the design estimator's percent bias, and clearance versus selection-robust top-five-percent capture and uplift as the band's covariate extent is varied. |
| `paper1_ignorability.py` | Post-clearance ignorability-violation sensitivity sweep, Table 6 (Section 6.2): leaking the unobserved confounder into the audit logit with strength lambda, the design estimator's percent bias and the hidden-selection tilt as ignorability fails, and the marginal-sensitivity-model breakdown the design arm needs to still cover the population value. |
| `paper1_coverage.py` | Conformal coverage and the calibrated-uncertainty layer (Section 5.4 and Section 6.1, 6.3, 6.4): the selection-valid coverage-by-stratum table (marginal versus Mondrian, on the clearance versus the post-clearance arm), the conformalized-quantile-regression risk-coverage curve and the deferred-duty abstention numbers, and the three uncertainty-layer stress tests, failing overlap on the blind band, small sample down to two thousand declarations, and noisy band stratification up to forty percent mislabeling. Pass `quick` as the first argument for a fast smoke check at reduced sample size and seed count. |
| `paper1_attribution.py` | Supplementary Figure S1 (Section S3): the global permutation feature importances of the Stage-1 duty-scoring model, and the assumption-free Manski lower bound (about 71 percent of the true population value). |

## Requirements

CPU only; CPython 3.11 or newer. Pinned versions (see `requirements.txt`):

```
numpy==2.0.2
scikit-learn==1.6.1
matplotlib==3.10.8
```

```bash
pip install -r requirements.txt
```

Each script is self-contained and prints its results. Run any of them directly:

```bash
python paper1_backbone.py
python paper1_policy.py
python paper1_prop6_verify.py
python paper1_adverse_grid.py
python paper1_consistency.py
python paper1_bandsize.py
python paper1_ignorability.py
python paper1_coverage.py        # add `quick` for a fast smoke check
python paper1_attribution.py
```

On a single CPU the backbone runs in roughly six to nine minutes at `N=60000` and twelve seeds; `paper1_policy.py` at its fifty-seed default and `paper1_coverage.py` at full scale (it fits cross-fitted estimators inside the overlap and small-sample stress sweeps) are the longest, the others are faster; `paper1_coverage.py quick` runs the same cross-fitted pipeline at reduced sample size and seed count and finishes in a few minutes rather than the full runtime. Library versions are pinned so the reported numbers reproduce exactly; on a different stack the qualitative results hold and the third or fourth significant digit may move.

## Environment and seeds

CPython 3.11 or newer (required by the pinned `numpy`, `scikit-learn`, and `matplotlib`); CPU only and OS-independent. All randomness is seeded explicitly inside each script with fixed base seeds (for example 1000 for the operating points, 3000 for the identification ladder, 4000 for the uplift split, and 5000 for the inspected-only check) offset by the replicate index, so every run is deterministic. Installing the pinned packages into a clean environment reproduces the reported values.

## Expected outputs

**`paper1_backbone.py`** (realistic configuration, `N=60000`, twelve seeds):

- Operating points: fraud prevalence about 3.1 percent, clearance inspection rate about 12.5 percent, hit rate among inspected about 5.1 percent, Gini of recoverable duty about 0.604; the blind band is about 23 percent of declarations and holds about 71 percent of recoverable duty.
- Identification ladder (percent error versus the true population value): naive about +20, clearance IPW about -51, clearance AIPW about -52, design (post-clearance) AIPW about +2.
- Held-out policy uplift (top five percent): clearance-trained capture about 44 percent, selection-robust capture about 61 percent, uplift about +40 percent at realistic and about +2 percent at benign.
- Inspected-only evaluation bias (top five percent): the field protocol scores the selection-robust policy about sixteen points below its true value (the headline figure is the fifty-seed estimate from `paper1_policy.py`; the twelve-seed backbone run reproduces it to within sampling noise).

**`paper1_policy.py`** (realistic and benign regimes, fifty seeds): in the realistic regime the selection-robust policy captures about 60.9 percent of recoverable duty at a top-five-percent budget against about 44.6 percent for the clearance-trained policy, a relative uplift of about +38.6 percent (ninety-five-percent confidence interval about plus or minus 5.4) by mean of ratios and about +36.3 percent by ratio of means. The budget sweep is strictly monotone decreasing, from about +54.9 percent at top one percent through about +38.6 percent at top five percent to about +26.0 percent at top twenty percent, so the advantage is largest where audit capacity is scarcest. In the benign regime the two policies are statistically indistinguishable (about +1.4 percent, confidence interval about plus or minus 2.7). The inspected-only evaluation understates the selection-robust policy's captured-duty share by about sixteen points (confidence interval about plus or minus 1.8) relative to the oracle.

**`paper1_prop6_verify.py`**: empirical and predicted standard-error ratios near one across band audit rates with negligible bias; finite ninety-percent intervals once the calibration count clears the informativeness floor; and at a representative target the derived rate is met exactly at the operating boundary and fails just below it, confirming the rule predicts the boundary rather than being fitted to it.

**`paper1_consistency.py`** (twenty seeds): from ten thousand declarations upward the percent bias stays within sampling noise of zero, the standard error scaled by the square root of the sample size holds near four (the root-n rate of Proposition 1), and the influence-function interval covers the population value in every seed; the five-thousand-declaration row shows a downward small-sample bias, wide across-seed dispersion, and one interval miss out of twenty, marking the marginal regime. Defaults to twenty seeds; pass an integer first argument to override.

**`paper1_adverse_grid.py`**: benign uplift about +2 percent; at the calibrated operating point the band holds about 70 percent of recoverable duty with clearance capture about 44 percent, selection-robust capture about 60 percent, and uplift about +38 percent; the uplift rises monotonically with the band's duty share (about +1 percent when the band holds only its proportional share, to about +88 percent when it holds most of the duty).

**`paper1_bandsize.py`**: as the blind band is widened from about a tenth to about two-fifths of declarations (concealing from about half to about five-sixths of recoverable duty), the design estimator stays essentially unbiased, the selection-robust top-five-percent capture holds near sixty percent, and the uplift over the clearance-trained policy stays large (about forty percent, rising toward forty-five as the band hides more duty). The twenty-three-percent row reproduces the calibrated operating point.

**`paper1_ignorability.py`**: at lambda equal to zero the design estimator's bias is within sampling noise of the +2 percent reported for the realistic regime; as the audit logit's dependence on the unobserved confounder rises, the hidden-selection tilt grows toward a full standard deviation and the point estimate's bias climbs monotonically to about +32 percent, yet the marginal-sensitivity-model bound taken at the clearance-arm breakdown (Gamma star about 1.65) covers the population value in every seed across the whole violation range, because the design arm itself needs only Gamma star about 1.06 to 1.34. The point estimate is sensitive to the assumption; the sensitivity interval the paper recommends reporting alongside it is not.

**`paper1_coverage.py`** (realistic configuration; full scale `N=40000` with six to eight seeds, nominal 90 percent):
- Part A, coverage by stratum (Section 5.4): the marginal interval calibrated on the population covers the blind (cold) band at only about 79 percent, while covering the routine region at about 94 percent; the Mondrian (stratum-conditional) interval calibrated on the post-clearance arm covers both strata near about 94 percent with roughly 1,250 to 1,300 band calibration declarations; the clearance arm yields zero band calibration declarations (no defined band interval, shown as a zero bar) and its marginal coverage falls to about 74 percent. These are the numbers in the coverage-by-stratum figure.
- Part B, risk-coverage and abstention (Section 5.4): the conformalized-quantile-regression intervals hold near-nominal stratum-conditional coverage; deferring the widest ten percent of declarations concentrates about 76 to 77 percent of recoverable duty in the deferred set, and deferring thirty percent raises the deferred share to about 92 percent. These are the numbers in the risk-coverage figure.
- Part C, failing overlap (Section 6.1): as the post-clearance audit probability on the band falls, the band calibration set collapses from hundreds of declarations toward single digits and the point estimate's bias explodes, while stratum-conditional cold-band coverage stays within sampling noise of nominal throughout (the overlap-breakdown figure).
- Part D, small sample (Section 6.3): the point estimate is unusable at two thousand declarations (large bias and across-seed dispersion) and becomes reliable from about ten thousand upward, while cold-band coverage stays near nominal at every sample size; intervals widen rather than losing validity.
- Part E, noisy stratification (Section 6.4): coverage on the true blind band holds within sampling noise of nominal through mislabeling of about one fifth of the band, and degrades only under gross misidentification (forty percent). On a different software stack the qualitative results hold and the third or fourth significant digit may move; the `quick` mode uses a smaller sample and fewer seeds and is indicative only.

## License

Code released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper. This reproducibility package is permanently archived on Zenodo:

Paul, S. (2026). *Protecting Customs Revenue with Scarce Audit Capacity: Selection-Robust Targeting of Post-Clearance Undervaluation under Label Bias: Reproducibility Package* (v1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20941357

The version DOI above resolves to the exact archived release used in the paper (v1.0.1). The concept DOI https://doi.org/10.5281/zenodo.20941356 always resolves to the latest version.

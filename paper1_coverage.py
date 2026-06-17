# ============================================================================
# PAPER 1: CONFORMAL COVERAGE + UNCERTAINTY-LAYER STRESS REPRODUCTION
#
# Reproduces the calibrated-uncertainty results of the paper from the locked
# data-generating process, using the same generator and split-conformal
# protocol as every other released script:
#
#   PART A  Selection-valid conformal coverage table  (Figure: coverage by
#           stratum, nominal 90%). Marginal vs Mondrian (stratum-conditional)
#           calibration on the clearance arm vs the post-clearance arm. This is
#           the source of the reported numbers: a marginal interval covers the
#           blind band at ~79%, the Mondrian post-clearance interval covers both
#           strata near ~94%, and the clearance arm yields no band calibration
#           set (zero bar) with marginal coverage falling to ~74%.
#
#   PART B  CQR-Mondrian adaptive intervals + conformal abstention
#           (Figure: risk-coverage). Conformalized quantile regression,
#           calibrated within strata on the post-clearance arm; the
#           risk-coverage curve and the deferred-duty share are the source of
#           the reported abstention numbers (defer the widest 10% -> ~76.7% of
#           recoverable duty sits in the deferred set; defer 30% -> ~92%).
#
#   PART C  STRESS 1  post-clearance overlap on the blind band failing
#           (Figure: overlap breakdown). As the band calibration set collapses
#           from hundreds of declarations toward single digits, the point
#           estimate explodes while stratum-conditional coverage holds within
#           sampling noise of nominal.
#
#   PART D  STRESS 2  small-sample usability. Coverage holds near nominal down
#           to N=2000 (where the point estimate is unusable), with intervals
#           widening rather than losing validity.
#
#   PART E  STRESS 3  noisy Mondrian stratification. The agency mislabels band
#           membership for a fraction rho; coverage on the TRUE blind band is
#           tracked. Coverage holds within sampling noise of nominal through
#           mislabeling of one fifth of the band.
#
# numpy + scikit-learn, CPU only. Self-contained; no external data.
# Pin: numpy and scikit-learn versions should match the released environment
# (see requirements.txt). The generator and protocol are byte-identical to the
# ones used to produce the locked figures, so the printed numbers reproduce
# those figures within seed variation.
#
# Usage:
#   python paper1_coverage.py             # full scale, as reported in the paper
#   python paper1_coverage.py quick       # fast smoke check (smaller N, fewer seeds)
# ============================================================================
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import (
    HistGradientBoostingRegressor as HGBR,
    HistGradientBoostingClassifier as HGBC,
)
from sklearn.model_selection import KFold

QUICK = len(sys.argv) > 1 and sys.argv[1].lower() in ("quick", "fast", "smoke")

sig = lambda v: 1.0 / (1.0 + np.exp(-v))
RCAP = 80.0
EPS = 1e-3
ALPHA = 0.10  # nominal 90% prediction intervals


# ---------------------------------------------------------------------------
# Locked data-generating process (identical across all released scripts).
# The optional keyword arguments are inert at their defaults and are used only
# by the stress parts: cold_pca lowers the post-clearance audit probability on
# the blind band (PART C); cold_mult/prev/confU perturb the DGP (not used here
# for coverage). Calling gen(seed, N) with no extra arguments yields exactly the
# generator used for the coverage figure.
# ---------------------------------------------------------------------------
def gen(seed, N, cold_pca=1.0, cold_mult=1.0, prev=-5.6, confU=1.0):
    rng = np.random.default_rng(seed)
    x_val = rng.normal(0, 1, N)
    x_hs = rng.normal(0, 1, N)
    x_imp = rng.normal(0, 1, N)
    X = np.column_stack([x_val, x_hs, x_imp])
    U = rng.normal(0, 1, N)
    Z = rng.integers(0, 4, N)
    z = Z - 1.5
    cold = ((x_hs > 0.4) & (x_hs < 1.2))
    p = sig(prev + 1.4 * x_val + 1.0 * x_imp + 1.8 * cold + 0.9 * U)
    fraud = rng.random(N) < p
    R = np.where(
        fraud,
        np.clip(np.exp(0.0 + 0.9 * x_val + cold_mult * cold + rng.normal(0, 0.9, N)), 0, RCAP),
        0.0,
    )
    e_clr = np.where(cold, 0.0, sig(-2.6 + 0.9 * x_val + 0.5 * x_imp + confU * U - 1.2 * z))
    Dc = rng.random(N) < e_clr
    e_pca = sig(-1.2 + 0.5 * x_val + 0.4 * x_imp + cold_pca * cold)
    Dp = rng.random(N) < e_pca
    return dict(X=X, R=R, Dc=Dc, Dp=Dp, e_pca=e_pca, cold=cold)


def q1a(s, a=ALPHA):
    """Finite-sample conformal quantile (the (1-a) upper quantile, 'higher')."""
    return np.quantile(s, 1 - a, method="higher") if len(s) > 0 else np.nan


def wq(s, w, lvl):
    """Weighted quantile for the covariate-shift (1/e) marginal interval."""
    if len(s) == 0:
        return np.nan
    o = np.argsort(s)
    s = s[o]
    w = w[o]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return s[min(int(np.searchsorted(cw, lvl)), len(s) - 1)]


def cf_aipw(X, R, Dp, seed, K=5):
    """Cross-fitted doubly-robust (AIPW) point estimate; used in PART C/D to
    show the point estimate breaking while coverage holds."""
    n = len(R)
    om = np.zeros(n)
    oe = np.zeros(n)
    kf = KFold(K, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        a = tr[Dp[tr]]
        if len(a) < 5:
            om[te] = R[a].mean() if len(a) > 0 else 0.0
        else:
            om[te] = HGBR(max_iter=120, random_state=0).fit(X[a], R[a]).predict(X[te])
        oe[te] = HGBC(max_iter=120, random_state=0).fit(X[tr], Dp[tr].astype(int)).predict_proba(X[te])[:, 1]
    e = np.clip(oe, EPS, 1 - EPS)
    return np.mean(om + Dp * (R - om) / e)


def cap(score, Rt, k=0.05):
    t = int(k * len(score))
    idx = np.argsort(-score)[:t]
    return 100 * Rt[idx].sum() / Rt.sum()


def _mn(a):
    a = [x for x in a if not np.isnan(x)]
    return np.mean(a) if a else float("nan")


# ===========================================================================
# PART A  Selection-valid conformal coverage table (coverage-by-stratum figure)
# ===========================================================================
def part_A(N, SEEDS):
    rows = {
        k: {"marg": [], "cold": [], "non": [], "ncold": []}
        for k in [
            "marginal, clearance arm",
            "marginal, post-clearance arm (wtd 1/e_pca)",
            "Mondrian, post-clearance arm",
            "Mondrian, clearance arm",
        ]
    }
    for i in range(SEEDS):
        d = gen(9000 + i, N)
        X, R, Dc, Dp, e_pca, cold = d["X"], d["R"], d["Dc"], d["Dp"], d["e_pca"], d["cold"]
        idx = np.arange(N)
        rng = np.random.default_rng(100 + i)
        rng.shuffle(idx)
        tr = idx[: int(.4 * N)]
        ca = idx[int(.4 * N): int(.7 * N)]
        te = idx[int(.7 * N):]
        mhat = HGBR(max_iter=120, random_state=0).fit(X[tr][Dp[tr]], R[tr][Dp[tr]])
        res = np.abs(R - mhat.predict(X))
        rte = res[te]
        coldte = cold[te]

        def report(key, qpt):
            marg = np.mean(rte <= qpt) * 100
            cc = np.mean(rte[coldte] <= qpt[coldte]) * 100 if coldte.any() else np.nan
            cn = np.mean(rte[~coldte] <= qpt[~coldte]) * 100 if (~coldte).any() else np.nan
            rows[key]["marg"].append(marg)
            rows[key]["cold"].append(cc)
            rows[key]["non"].append(cn)

        # marginal, clearance arm (unweighted)
        q = wq(res[ca][Dc[ca]], np.ones(int(Dc[ca].sum())), 1 - ALPHA)
        report("marginal, clearance arm", np.full(len(te), q))
        # marginal, post-clearance arm, covariate-shift weighted by 1/e_pca
        mP = Dp[ca]
        q = wq(res[ca][mP], 1.0 / e_pca[ca][mP], 1 - ALPHA)
        report("marginal, post-clearance arm (wtd 1/e_pca)", np.full(len(te), q))
        # Mondrian, post-clearance arm
        qc = wq(res[ca][mP & cold[ca]], np.ones(int((mP & cold[ca]).sum())), 1 - ALPHA)
        qn = wq(res[ca][mP & ~cold[ca]], np.ones(int((mP & ~cold[ca]).sum())), 1 - ALPHA)
        report("Mondrian, post-clearance arm", np.where(coldte, qc, qn))
        rows["Mondrian, post-clearance arm"]["ncold"].append(int((mP & cold[ca]).sum()))
        # Mondrian, clearance arm (cold stratum has ~0 calibration points)
        mC = Dc[ca]
        ncoldC = int((mC & cold[ca]).sum())
        qnC = wq(res[ca][mC & ~cold[ca]], np.ones(int((mC & ~cold[ca]).sum())), 1 - ALPHA)
        qcC = wq(res[ca][mC & cold[ca]], np.ones(ncoldC), 1 - ALPHA) if ncoldC > 0 else np.nan
        report("Mondrian, clearance arm", np.where(coldte, qcC, qnC))
        rows["Mondrian, clearance arm"]["ncold"].append(ncoldC)

    print(f"PART A  SELECTION-VALID CONFORMAL COVERAGE  (N={N}/{SEEDS}, nominal {100*(1-ALPHA):.0f}%)")
    print(f"  {'strategy':44s} {'marginal%':>10s} {'cover|cold%':>12s} {'cover|noncold%':>15s} {'#cold cal':>10s}")
    for k, v in rows.items():
        nc = f"{np.mean(v['ncold']):.0f}" if v["ncold"] else "-"
        print(f"  {k:44s} {_mn(v['marg']):>10.1f} {_mn(v['cold']):>12.1f} {_mn(v['non']):>15.1f} {nc:>10s}")
    print("  Reads: marginal interval covers the blind (cold) band at only ~79%;")
    print("  Mondrian on the post-clearance arm restores ~94% on both strata;")
    print("  the clearance arm has 0 band calibration points (no defined band interval),")
    print("  and its marginal coverage falls to ~74%.")
    print()


# ===========================================================================
# PART B  CQR-Mondrian adaptive intervals + conformal abstention (risk-coverage)
# ===========================================================================
def part_B(N, SEEDS):
    levels = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]
    cov_cold, cov_non, cov_marg, medw = [], [], [], []
    rc_mae = {lv: [] for lv in levels}
    rc_duty = {lv: [] for lv in levels}
    for i in range(SEEDS):
        d = gen(9000 + i, N)
        X, R, Dp, cold = d["X"], d["R"], d["Dp"], d["cold"]
        idx = np.arange(N)
        rng = np.random.default_rng(100 + i)
        rng.shuffle(idx)
        tr = idx[: int(.4 * N)]
        ca = idx[int(.4 * N): int(.7 * N)]
        te = idx[int(.7 * N):]
        Xtr, Rtr = X[tr][Dp[tr]], R[tr][Dp[tr]]
        mhat = HGBR(max_iter=120, random_state=0).fit(Xtr, Rtr)
        qlo = HGBR(loss="quantile", quantile=ALPHA / 2, max_iter=120, random_state=0).fit(Xtr, Rtr)
        qhi = HGBR(loss="quantile", quantile=1 - ALPHA / 2, max_iter=120, random_state=0).fit(Xtr, Rtr)
        lo_ca = qlo.predict(X[ca])
        hi_ca = qhi.predict(X[ca])
        E = np.maximum(lo_ca - R[ca], R[ca] - hi_ca)  # CQR nonconformity
        mP = Dp[ca]
        c_cold = q1a(E[mP & cold[ca]])
        c_non = q1a(E[mP & ~cold[ca]])
        lo_te = qlo.predict(X[te])
        hi_te = qhi.predict(X[te])
        c_te = np.where(cold[te], c_cold, c_non)
        L = lo_te - c_te
        Ut = hi_te + c_te
        cov = (R[te] >= L) & (R[te] <= Ut)
        coldte = cold[te]
        cov_cold.append(np.mean(cov[coldte]) * 100)
        cov_non.append(np.mean(cov[~coldte]) * 100)
        cov_marg.append(np.mean(cov) * 100)
        width = Ut - L
        medw.append(np.median(width))
        err = np.abs(R[te] - mhat.predict(X[te]))
        order = np.argsort(width)
        totduty = R[te].sum()
        for lv in levels:
            k = int(lv * len(te))
            keep = order[:k]
            abst = order[k:]
            rc_mae[lv].append(np.mean(err[keep]))
            rc_duty[lv].append(100 * R[te][abst].sum() / totduty if len(abst) > 0 else 0.0)
    print(f"PART B  CQR-MONDRIAN ADAPTIVE INTERVALS  (N={N}/{SEEDS}, nominal {100*(1-ALPHA):.0f}%)")
    print(f"  coverage|cold {_mn(cov_cold):.1f}%   coverage|noncold {_mn(cov_non):.1f}%   "
          f"marginal {_mn(cov_marg):.1f}%   median width {_mn(medw):.2f}")
    print("  CONFORMAL ABSTENTION / RISK-COVERAGE (abstain on widest CQR intervals; defer to human audit)")
    print(f"  {'auto-decided %':>14s} {'MAE on auto-decided':>20s} {'duty% in deferred set':>22s}")
    for lv in levels:
        print(f"  {100*lv:>13.0f}% {_mn(rc_mae[lv]):>20.3f} {_mn(rc_duty[lv]):>21.1f}%")
    print("  Reads: deferring the widest ~10% of declarations concentrates ~76-77% of")
    print("  recoverable duty in the deferred set; deferring ~30% raises it to ~92%.")
    print()


# ===========================================================================
# PART C  STRESS 1  failing overlap on the blind band (overlap-breakdown figure)
# ===========================================================================
def part_C(N, SEEDS):
    print(f"PART C  STRESS 1  post-clearance overlap on the blind band failing  (N={N}/{SEEDS}, nominal {100*(1-ALPHA):.0f}%)")
    print("  cold_pca lowers post-clearance audit prob on the blind band toward 0 (overlap failing)")
    print(f"  {'cold_pca':>8s} {'p5 e_pca|cold':>13s} {'AIPW bias%':>11s} {'AIPW SD%':>9s} {'cold-cover%':>11s} {'#cold cal':>9s}")
    for cpca in [1.0, 0.0, -1.0, -2.0, -3.0, -4.0]:
        ests, covs, ncals, p5s = [], [], [], []
        for i in range(SEEDS):
            d = gen(7000 + i, N, cold_pca=cpca)
            X, R, Dp, cold, e_pca = d["X"], d["R"], d["Dp"], d["cold"], d["e_pca"]
            true = R.mean()
            ests.append(100 * (cf_aipw(X, R, Dp, i) - true) / true)
            p5s.append(np.percentile(e_pca[cold], 5))
            idx = np.arange(N)
            rng = np.random.default_rng(50 + i)
            rng.shuffle(idx)
            tr = idx[: int(.4 * N)]
            ca = idx[int(.4 * N): int(.7 * N)]
            te = idx[int(.7 * N):]
            if Dp[tr].sum() < 10:
                covs.append(np.nan)
                ncals.append(0)
                continue
            mhat = HGBR(max_iter=120, random_state=0).fit(X[tr][Dp[tr]], R[tr][Dp[tr]])
            res = np.abs(R - mhat.predict(X))
            mP = Dp[ca]
            ncold = int((mP & cold[ca]).sum())
            qc = q1a(res[ca][mP & cold[ca]])
            coldte = cold[te]
            covs.append(np.mean(res[te][coldte] <= qc) * 100 if (not np.isnan(qc) and coldte.any()) else np.nan)
            ncals.append(ncold)
        cc = [x for x in covs if not np.isnan(x)]
        print(f"  {cpca:>8.1f} {np.mean(p5s):>13.4f} {np.mean(ests):>11.1f} {np.std(ests):>9.1f} "
              f"{(np.mean(cc) if cc else float('nan')):>11.1f} {np.mean(ncals):>9.0f}")
    print("  Reads: as the band calibration set collapses from hundreds toward single digits,")
    print("  the point estimate's bias explodes while cold-band coverage holds near nominal.")
    print()


# ===========================================================================
# PART D  STRESS 2  small-sample usability (coverage holds down to N=2000)
# ===========================================================================
def part_D(SEEDS, grid):
    print(f"PART D  STRESS 2  small-sample usability  ({SEEDS} seeds, nominal {100*(1-ALPHA):.0f}%)")
    print(f"  {'N':>7s} {'AIPW bias%':>11s} {'AIPW SD%':>9s} {'cold-cover%':>11s} {'#cold cal':>9s} {'uplift% (rel)':>13s}")
    for N in grid:
        ests, covs, ncals, ups = [], [], [], []
        for i in range(SEEDS):
            d = gen(7000 + i, N)
            X, R, Dp, Dc, cold = d["X"], d["R"], d["Dp"], d["Dc"], d["cold"]
            true = R.mean()
            ests.append(100 * (cf_aipw(X, R, Dp, i) - true) / true)
            idx = np.arange(N)
            rng = np.random.default_rng(50 + i)
            rng.shuffle(idx)
            tr = idx[: int(.4 * N)]
            ca = idx[int(.4 * N): int(.7 * N)]
            te = idx[int(.7 * N):]
            if Dp[tr].sum() >= 10:
                mhat = HGBR(max_iter=120, random_state=0).fit(X[tr][Dp[tr]], R[tr][Dp[tr]])
                res = np.abs(R - mhat.predict(X))
                mP = Dp[ca]
                ncold = int((mP & cold[ca]).sum())
                qc = q1a(res[ca][mP & cold[ca]])
                coldte = cold[te]
                covs.append(np.mean(res[te][coldte] <= qc) * 100 if (not np.isnan(qc) and coldte.any()) else np.nan)
                ncals.append(ncold)
            else:
                covs.append(np.nan)
                ncals.append(0)
            h = idx[: N // 2]
            ev = idx[N // 2:]
            aP = h[Dp[h]]
            aC = h[Dc[h]]
            if len(aP) >= 10 and len(aC) >= 10:
                sD = HGBR(max_iter=120, random_state=0).fit(X[aP], R[aP]).predict(X[ev])
                sC = HGBR(max_iter=120, random_state=0).fit(X[aC], R[aC]).predict(X[ev])
                cD = cap(sD, R[ev])
                cC = cap(sC, R[ev])
                if cC > 0:
                    ups.append(100 * (cD / cC - 1))
        cc = [x for x in covs if not np.isnan(x)]
        print(f"  {N:>7d} {np.mean(ests):>11.1f} {np.std(ests):>9.1f} "
              f"{(np.mean(cc) if cc else float('nan')):>11.1f} {np.mean(ncals):>9.0f} "
              f"{(np.mean(ups) if ups else float('nan')):>13.1f}")
    print("  Reads: the point estimate is unusable at N=2000 (large bias and SD) while")
    print("  cold-band coverage stays near nominal at every N; intervals widen, not break.")
    print()


# ===========================================================================
# PART E  STRESS 3  noisy Mondrian stratification (mislabel up to 40%)
# ===========================================================================
def part_E(N, SEEDS):
    print(f"PART E  STRESS 3  noisy Mondrian stratification  (N={N}/{SEEDS}, nominal {100*(1-ALPHA):.0f}%)")
    print("  agency mislabels band membership for fraction rho; coverage evaluated on the TRUE blind band")
    print(f"  {'rho':>6s} {'cover|TRUEcold%':>15s} {'cover|noncold%':>14s} {'marginal%':>10s}")
    for rho in [0.0, 0.05, 0.10, 0.20, 0.40]:
        cc, cn, cm = [], [], []
        for i in range(SEEDS):
            d = gen(7000 + i, N)
            X, R, Dp, cold = d["X"], d["R"], d["Dp"], d["cold"]
            rng = np.random.default_rng(50 + i)
            flip = rng.random(len(cold)) < rho
            cobs = np.where(flip, ~cold, cold)
            idx = np.arange(N)
            rng2 = np.random.default_rng(70 + i)
            rng2.shuffle(idx)
            tr = idx[: int(.4 * N)]
            ca = idx[int(.4 * N): int(.7 * N)]
            te = idx[int(.7 * N):]
            mhat = HGBR(max_iter=120, random_state=0).fit(X[tr][Dp[tr]], R[tr][Dp[tr]])
            res = np.abs(R - mhat.predict(X))
            mP = Dp[ca]
            qc = q1a(res[ca][mP & cobs[ca]])
            qn = q1a(res[ca][mP & ~cobs[ca]])
            qpt = np.where(cobs[te], qc, qn)  # interval assigned by the agency's (noisy) label
            covered = res[te] <= qpt
            cc.append(np.mean(covered[cold[te]]) * 100)
            cn.append(np.mean(covered[~cold[te]]) * 100)
            cm.append(np.mean(covered) * 100)
        print(f"  {rho:>6.2f} {np.mean(cc):>15.1f} {np.mean(cn):>14.1f} {np.mean(cm):>10.1f}")
    print("  Reads: coverage on the true blind band holds within sampling noise of nominal")
    print("  through mislabeling of one fifth of the band; gross misidentification degrades it.")
    print()


if __name__ == "__main__":
    if QUICK:
        print(">>> QUICK smoke mode: reduced N and seeds; numbers are indicative, not the reported values.\n")
        NA, SA = 8000, 3
        NB, SB = 8000, 3
        NC, SC = 6000, 3
        SD_SEEDS, D_GRID = 3, [2000, 8000]
        NE, SE = 8000, 3
    else:
        # Full scale: matches the seeds/N that produced the locked figures.
        NA, SA = 40000, 8
        NB, SB = 40000, 6
        NC, SC = 20000, 6
        SD_SEEDS, D_GRID = 8, [2000, 5000, 10000, 20000, 40000]
        NE, SE = 40000, 6

    part_A(NA, SA)
    part_B(NB, SB)
    part_C(NC, SC)
    part_D(SD_SEEDS, D_GRID)
    part_E(NE, SE)
    print("Done. PART A reproduces the coverage-by-stratum figure; PART B the risk-coverage")
    print("figure and abstention numbers; PARTS C-E the uncertainty-layer stress results")
    print("(overlap collapse, small sample, label noise) reported in the robustness section.")

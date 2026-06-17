# ============================================================================
# PAPER 1: POLICY-VALUE REPRODUCTION (held-out uplift, budget sweep,
# benign control, inspected-only evaluation bias). Mirrors the C and D blocks
# of paper1_backbone.py at higher seed count for the reported confidence
# intervals. numpy/scikit-learn, CPU. Default 50 seeds; override via argv[1].
#   python paper1_policy.py            # 50 seeds (as reported in the paper)
#   python paper1_policy.py 12         # quick check
# ============================================================================
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

N = 60000
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
RCAP = 80.0
RK = dict(max_iter=120, random_state=0)
sig = lambda v: 1 / (1 + np.exp(-v))


def gen(seed, N, confound, cold_on):                 # locked calibrated DGP (== backbone)
    rng = np.random.default_rng(seed)
    x_val = rng.normal(0, 1, N); x_hs = rng.normal(0, 1, N); x_imp = rng.normal(0, 1, N)
    X = np.column_stack([x_val, x_hs, x_imp]); U = rng.normal(0, 1, N)
    Z = rng.integers(0, 4, N); z = Z - 1.5
    cold = ((x_hs > 0.4) & (x_hs < 1.2)) if cold_on else np.zeros(N, bool)
    p = sig(-5.6 + 1.4 * x_val + 1.0 * x_imp + 1.8 * cold + (0.9 if confound else 0.0) * U)
    fraud = rng.random(N) < p
    R = np.where(fraud, np.clip(np.exp(0.0 + 0.9 * x_val + 1.0 * cold + rng.normal(0, 0.9, N)), 0, RCAP), 0.0)
    Dc = rng.random(N) < np.where(cold, 0.0, sig(-2.6 + 0.9 * x_val + 0.5 * x_imp + (1.0 if confound else 0.0) * U - 1.2 * z))
    Dp = rng.random(N) < sig(-1.2 + 0.5 * x_val + 0.4 * x_imp + 1.0 * cold)
    Db = rng.random(N) < 0.20
    return dict(X=X, R=R, Dc=Dc, Dp=Dp, Db=Db, cold=cold, fraud=fraud)


def cap(score, Rtrue, k):                             # share of true recoverable duty in top-k by score
    n = len(score); pick = np.argsort(-score)[:max(1, int(k * n))]; tot = Rtrue.sum()
    return (Rtrue[pick].sum() / tot) if tot > 0 else 0.0


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return 1.96 * a.std() / np.sqrt(len(a))


print(f"==== POLICY VALUE  N={N}  SEEDS={SEEDS} ====")

# C. held-out policy uplift, top-5% (50/50 disjoint train/test)
# clearance-trained score sn (fit on clearance-audited rows) vs design score sr
# (fit on post-clearance-audited rows); evaluated by true recoverable-duty capture.
def uplift(tag, cf, cold_on, k=0.05):
    nc = []; rc = []; ups = []
    for i in range(SEEDS):
        d = gen(4000 + i, N, cf, cold_on); X = d['X']; R = d['R']; Dc = d['Dc']; Dp = d['Dp']
        idx = np.arange(len(X)); rng = np.random.default_rng(10 + i); rng.shuffle(idx)
        tr = idx[:len(X) // 2]; te = idx[len(X) // 2:]
        sn = HGBR(**RK).fit(X[tr][Dc[tr]], R[tr][Dc[tr]]).predict(X[te]) if Dc[tr].sum() > 20 else np.zeros(len(te))
        sr = HGBR(**RK).fit(X[tr][Dp[tr]], R[tr][Dp[tr]]).predict(X[te]) if Dp[tr].sum() > 20 else np.zeros(len(te))
        cN = cap(sn, R[te], k); cR = cap(sr, R[te], k)
        nc.append(cN * 100); rc.append(cR * 100)
        ups.append(100 * (cR - cN) / cN if cN > 0 else np.nan)
    mor = np.nanmean(ups)                                   # mean of per-seed ratios
    rom = 100 * (np.mean(rc) - np.mean(nc)) / np.mean(nc)   # ratio of means
    print(f"  [{tag:10s}] clearance cap {np.mean(nc):4.1f}%  design cap {np.mean(rc):4.1f}%  "
          f"uplift(MoR) {mor:+5.1f}+/-{ci(ups):.1f}  uplift(RoM) {rom:+5.1f}")
    return mor

uplift("REALISTIC", True, True)
uplift("BENIGN", False, False)   # control: gain should vanish when band carries no excess duty

# budget sweep (REALISTIC), same draws, varying top-k
print("BUDGET SWEEP (REALISTIC, mean-of-ratios uplift):")
for k in (0.01, 0.02, 0.05, 0.10, 0.20):
    ups = []
    for i in range(SEEDS):
        d = gen(4000 + i, N, True, True); X = d['X']; R = d['R']; Dc = d['Dc']; Dp = d['Dp']
        idx = np.arange(len(X)); rng = np.random.default_rng(10 + i); rng.shuffle(idx)
        tr = idx[:len(X) // 2]; te = idx[len(X) // 2:]
        sn = HGBR(**RK).fit(X[tr][Dc[tr]], R[tr][Dc[tr]]).predict(X[te]) if Dc[tr].sum() > 20 else np.zeros(len(te))
        sr = HGBR(**RK).fit(X[tr][Dp[tr]], R[tr][Dp[tr]]).predict(X[te]) if Dp[tr].sum() > 20 else np.zeros(len(te))
        cN = cap(sn, R[te], k); cR = cap(sr, R[te], k)
        ups.append(100 * (cR - cN) / cN if cN > 0 else np.nan)
    print(f"  top-{int(k*100):2d}%  uplift {np.nanmean(ups):+5.1f}+/-{ci(ups):.1f}")

# D. inspected-only evaluation bias (field-minus-oracle, top-5%)
infN = []; infR = []
for i in range(SEEDS):
    d = gen(5000 + i, N, True, True); X = d['X']; R = d['R']; Dc = d['Dc']; Dp = d['Dp']
    idx = np.arange(len(X)); rng = np.random.default_rng(20 + i); rng.shuffle(idx)
    tr = idx[:len(X) // 2]; te = idx[len(X) // 2:]
    sn = HGBR(**RK).fit(X[tr][Dc[tr]], R[tr][Dc[tr]]).predict(X[te])
    sr = HGBR(**RK).fit(X[tr][Dp[tr]], R[tr][Dp[tr]]).predict(X[te])
    Rte = R[te]; ins = Dc[te]
    infN.append((cap(sn[ins], Rte[ins], .05) - cap(sn, Rte, .05)) * 100)
    infR.append((cap(sr[ins], Rte[ins], .05) - cap(sr, Rte, .05)) * 100)
print(f"D. INSPECTED-ONLY EVALUATION BIAS (field-minus-oracle, top-5%): "
      f"naive {np.mean(infN):+.1f}+/-{ci(infN):.1f} pts   "
      f"selection-robust {np.mean(infR):+.1f}+/-{ci(infR):.1f} pts")
print("DONE")

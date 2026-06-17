# ============================================================================
# PAPER 1: auditability + assumption-free-bound check (standalone).
# (1) Confirms the Manski lower bound on recoverable duty equals about 71% of
#     the true population value in the realistic regime (assumption-free floor).
# (2) Fits the Stage-1 selection-robust duty-scoring model (gradient-boosted
#     regression on post-clearance-audited rows) and reports permutation
#     feature importances over the three declaration features, demonstrating
#     the auditability the paper claims for the transparent tree model.
#     Saves fig5_feature_attribution.pdf.
# numpy / scikit-learn / matplotlib, CPU only. Runs in 1-3 minutes.
# DGP is reproduced verbatim from paper1_backbone.py so this file is
# self-contained and does not trigger the backbone run on import.
# ============================================================================
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 60000
RCAP = 80.0
RK = dict(max_iter=120, random_state=0)
sig = lambda v: 1 / (1 + np.exp(-v))
FEATS = ["Valuation anomaly", "Tariff line (HS)", "Importer history"]


def gen(seed, N, confound, cold_on):                  # locked calibrated DGP (== backbone)
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


# ---------------------------------------------------------------------------
# (1) Assumption-free Manski lower bound as a fraction of the true value.
#     Observed = audited by either arm (clearance OR post-clearance).
#     Lower bound sets the unobserved recoverable duty to zero.
# ---------------------------------------------------------------------------
print("==== (1) MANSKI LOWER BOUND (realistic regime, % of true recoverable duty) ====")
fracs = []
for i in range(8):
    d = gen(3000 + i, N, True, True)
    R = d["R"]; obs = d["Dc"] | d["Dp"]
    truth = R.mean()
    lo = (R * obs).sum() / len(R)                      # unobserved set to 0
    fracs.append(100 * lo / truth)
print(f"  observed-arm coverage = clearance OR post-clearance")
print(f"  Manski lower bound = {np.mean(fracs):.1f}% +/- {1.96*np.std(fracs)/np.sqrt(len(fracs)):.1f}% of the true value")
print(f"  (manuscript states about 71%)")

# ---------------------------------------------------------------------------
# (2) Auditability: permutation importances of the Stage-1 duty-scoring model.
#     Model = HGBR fit on post-clearance-audited rows predicting recoverable
#     duty R, exactly the selection-robust scorer used for targeting.
#     HistGradientBoosting has no impurity importances, so we report
#     permutation importance (model-agnostic, computed on held-out data).
# ---------------------------------------------------------------------------
print("\n==== (2) FEATURE ATTRIBUTION (Stage-1 duty-scoring model, realistic regime) ====")
d = gen(4000, N, True, True)
X = d["X"]; R = d["R"]; Dp = d["Dp"]
Xa, Ra = X[Dp], R[Dp]                                  # post-clearance-audited rows
Xtr, Xte, Rtr, Rte = train_test_split(Xa, Ra, test_size=0.4, random_state=0)
model = HGBR(**RK).fit(Xtr, Rtr)
print(f"  audited rows used = {len(Xa)}  (train {len(Xtr)} / test {len(Xte)})")
print(f"  held-out R^2 = {model.score(Xte, Rte):.3f}")
pi = permutation_importance(model, Xte, Rte, n_repeats=10, random_state=0, n_jobs=-1)
order = np.argsort(pi.importances_mean)[::-1]
print("  permutation importance (drop in R^2 when feature is shuffled):")
for j in order:
    print(f"    {FEATS[j]:20s}  {pi.importances_mean[j]:.4f} +/- {pi.importances_std[j]:.4f}")

# ---- figure ----
o = np.argsort(pi.importances_mean)                    # ascending for barh
fig, ax = plt.subplots(figsize=(6.2, 2.8))
ax.barh([FEATS[j] for j in o], pi.importances_mean[o],
        xerr=pi.importances_std[o], color="#3b6ea5", ecolor="#444", capsize=3)
ax.set_xlabel("Permutation importance (mean decrease in held-out $R^2$)")
ax.set_title("Stage-1 duty-scoring model: feature attribution")
ax.margins(y=0.15)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("fig5_feature_attribution.pdf", bbox_inches="tight")
print("\n  saved fig5_feature_attribution.pdf")
print("DONE")

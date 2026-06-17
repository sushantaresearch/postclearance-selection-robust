# ============================================================================
# PAPER 1: FINITE-SAMPLE CONSISTENCY (Table 4): percent bias, SE x sqrt(N),
# and 95% CI coverage of the design PCA-AIPW estimator across sample sizes.
# Confirms the root-n rate of Proposition 1. numpy/scikit-learn, CPU.
# Usage: python paper1_consistency.py [n_seeds=20] [N_csv=all]
# ============================================================================
import numpy as np, sys
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import KFold
EPS=1e-3; RK=dict(max_iter=120,random_state=0); RCAP=80.0; sig=lambda v:1/(1+np.exp(-v))
SEEDS=int(sys.argv[1]) if len(sys.argv)>1 else 20
NS=[int(x) for x in sys.argv[2].split(",")] if len(sys.argv)>2 else [5000,10000,20000,40000,80000]
def gen(seed,N,confound=True,cold_on=True):
    rng=np.random.default_rng(seed)
    x_val=rng.normal(0,1,N); x_hs=rng.normal(0,1,N); x_imp=rng.normal(0,1,N)
    X=np.column_stack([x_val,x_hs,x_imp]); U=rng.normal(0,1,N); Z=rng.integers(0,4,N); z=Z-1.5
    cold=((x_hs>0.4)&(x_hs<1.2)) if cold_on else np.zeros(N,bool)
    p=sig(-5.6+1.4*x_val+1.0*x_imp+1.8*cold+(0.9 if confound else 0.0)*U); fraud=rng.random(N)<p
    R=np.where(fraud,np.clip(np.exp(0.0+0.9*x_val+1.0*cold+rng.normal(0,0.9,N)),0,RCAP),0.0)
    Dp=rng.random(N)<sig(-1.2+0.5*x_val+0.4*x_imp+1.0*cold)
    return X,R,Dp
def cf_clf(X,D,s):
    p=np.zeros(len(X))
    for tr,te in KFold(5,shuffle=True,random_state=s).split(X):
        p[te]=HGBC(**RK).fit(X[tr],D[tr].astype(int)).predict_proba(X[te])[:,1]
    return np.clip(p,EPS,1-EPS)
def cf_reg(X,mask,R,s):
    p=np.zeros(len(X))
    for tr,te in KFold(5,shuffle=True,random_state=s).split(X):
        m=mask[tr]; p[te]=HGBR(**RK).fit(X[tr][m],R[tr][m]).predict(X[te]) if m.sum()>20 else 0.0
    return p
print(f"=== CONSISTENCY SWEEP (design PCA-AIPW, realistic, {SEEDS} seeds) ===")
print(" N        %bias        SExsqrtN      95%-CI coverage")
for N in NS:
    bs=[];sds=[];cov=0
    for i in range(SEEDS):
        X,R,Dp=gen(3000+i,N); truth=R.mean()
        mpca=cf_reg(X,Dp,R,i); epca=cf_clf(X,Dp,i)
        phi=mpca+Dp*(R-mpca)/epca; est=phi.mean()
        se=phi.std(ddof=1)/np.sqrt(N)
        bs.append(100*(est-truth)/truth); sds.append(phi.std(ddof=1))
        cov+=(est-1.96*se<=truth<=est+1.96*se)
    print(f" {N:6d}   {np.mean(bs):+6.1f}+/-{1.96*np.std(bs)/np.sqrt(SEEDS):4.1f}   {np.mean(sds):5.2f}        {cov}/{SEEDS}")
print("Expect: |bias| shrinks toward 0; SExsqrtN ~constant near 4 (root-n); coverage ~nominal.")

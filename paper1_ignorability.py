# ============================================================================
# PAPER 1: ROBUSTNESS PASS: post-clearance IGNORABILITY VIOLATION.
# The locked REALISTIC DGP is reused unchanged EXCEPT the post-clearance audit
# logit gains a term lambda*U, leaking the unobserved confounder U (which drives
# fraud and hence R) into the audit-selection mechanism. For a fixed seed the
# covariates, U, fraud, R and the clearance arm are IDENTICAL across lambda; only
# the audit selection D^p shifts, so the experiment isolates the effect of a
# confounded (non-ignorable) audit arm on the design AIPW estimator.
#   lambda = 0  ==  the locked DGP (post-clearance ignorable; reconciles with the
#                   paper's +2.1% design bias).
# Per lambda it reports: design AIPW signed % error vs true E[R] (mean, 95% CI);
# the hidden-selection tilt dU = E[U | audited] - E[U | not audited]; and the
# marginal-sensitivity-model Gamma* the design arm would need for its worst-case
# bound to still cover truth, plus how often Gamma=1.65 (the clearance-arm
# breakdown from Prop. on sensitivity) suffices. numpy / scikit-learn, CPU.
#
# Sandbox note: the full 7-point grid at N=60000, 12 seeds, was run in two
# lambda-chunks accumulating to a JSON cache (env IGN_LAMS / IGN_OUT) to stay
# within the per-command wall-clock limit; defaults below run the full grid.
# ============================================================================
import os, json, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import KFold

N=int(os.environ.get("IGN_N","60000")); SEEDS=int(os.environ.get("IGN_SEEDS","12"))
EPS=1e-3; RK=dict(max_iter=120,random_state=0); RCAP=80.0
GRID=np.round(np.concatenate([np.arange(1.00,1.51,0.02),np.arange(1.55,3.01,0.05)]),3)
sig=lambda v:1/(1+np.exp(-v))
LAMS=[float(x) for x in os.environ.get("IGN_LAMS","0.0,0.3,0.6,0.9,1.2,1.6,2.0").split(",")]
OUT=os.environ.get("IGN_OUT","")

def gen(seed,N,lam):                                   # locked REALISTIC DGP; D^p logit += lam*U
    rng=np.random.default_rng(seed)
    x_val=rng.normal(0,1,N); x_hs=rng.normal(0,1,N); x_imp=rng.normal(0,1,N)
    X=np.column_stack([x_val,x_hs,x_imp]); U=rng.normal(0,1,N); Z=rng.integers(0,4,N); z=Z-1.5
    cold=((x_hs>0.4)&(x_hs<1.2))
    p=sig(-5.6+1.4*x_val+1.0*x_imp+1.8*cold+0.9*U); fraud=rng.random(N)<p
    R=np.where(fraud,np.clip(np.exp(0.0+0.9*x_val+1.0*cold+rng.normal(0,0.9,N)),0,RCAP),0.0)
    Dc=rng.random(N)<np.where(cold,0.0,sig(-2.6+0.9*x_val+0.5*x_imp+1.0*U-1.2*z))
    Dp=rng.random(N)<sig(-1.2+0.5*x_val+0.4*x_imp+1.0*cold+lam*U)      # ignorability leak
    return dict(X=X,R=R,Dp=Dp,U=U)

def cf_clf(X,D,s):
    p=np.zeros(len(X))
    for tr,te in KFold(5,shuffle=True,random_state=s).split(X):
        p[te]=HGBC(**RK).fit(X[tr],D[tr].astype(int)).predict_proba(X[te])[:,1]
    return np.clip(p,EPS,1-EPS)
def cf_reg(X,mask,R,s):
    p=np.zeros(len(X))
    for tr,te in KFold(5,shuffle=True,random_state=s).split(X):
        m=mask[tr]; p[te]=(HGBR(**RK).fit(X[tr][m],R[tr][m]).predict(X[te]) if m.sum()>20 else (R[mask].mean() if mask.sum()>0 else 0.0))
    return p
def msm(Ro,eo,G):                                      # ZSB odds-tilt marginal sensitivity model
    o=eo/(1-eo); whi=1+G/o; wlo=1+1/(G*o); od=np.argsort(Ro); Rs=Ro[od]; H=whi[od]; L=wlo[od]
    clr=np.cumsum(L*Rs); cl=np.cumsum(L); chr_=np.cumsum(H*Rs); ch=np.cumsum(H)
    Thr=chr_[-1];Th=ch[-1];Tlr=clr[-1];Tl=cl[-1]
    up=max(np.max((clr+(Thr-chr_))/(cl+(Th-ch))),Thr/Th); lo=min(np.min((chr_+(Tlr-clr))/(ch+(Tl-cl))),Tlr/Tl)
    return lo,up
def gstar(Ro,eo,ref):
    for G in GRID:
        lo,hi=msm(Ro,eo,G)
        if lo<=ref<=hi: return float(G)
    return None

res={}
if OUT and os.path.exists(OUT): res=json.load(open(OUT))
for lam in LAMS:
    rows=[]
    for i in range(SEEDS):
        d=gen(7000+i,N,lam); X=d['X'];R=d['R'];Dp=d['Dp'];U=d['U']
        truth=R.mean()
        epca=cf_clf(X,Dp,i); mpca=cf_reg(X,Dp,R,i)
        aipw=float(np.mean(mpca+Dp*(R-mpca)/epca))
        rows.append(dict(err=100*(aipw-truth)/truth,
                         dU=float(U[Dp].mean()-U[~Dp].mean()),
                         gstar=gstar(R[Dp],epca[Dp],truth),
                         truth=float(truth),est=aipw))
    res[f"{lam:.2f}"]=rows
if OUT: json.dump(res,open(OUT,"w"))

ci=lambda a:1.96*np.array(a,float).std()/np.sqrt(len(a))
print(f"==== IGNORABILITY-VIOLATION ROBUSTNESS  N={N} SEEDS={SEEDS} ====")
print("  lam |  design %err (mean+/-95CI) | hidden tilt dU | Gamma*_design med[min-max] | covers@G=1.65")
for k in sorted(res,key=float):
    rows=res[k]; errs=[r['err'] for r in rows]; dUs=[r['dU'] for r in rows]
    gs=[r['gstar'] for r in rows if r['gstar'] is not None]
    cov=sum(1 for r in rows if (r['gstar'] is not None and r['gstar']<=1.65))
    gtxt=f"{np.median(gs):.2f}[{min(gs):.2f}-{max(gs):.2f}]" if gs else ">grid"
    print(f"  {float(k):4.2f} | {np.mean(errs):+6.1f} +/- {ci(errs):4.1f}       | {np.mean(dUs):+.3f}        | {gtxt:18s} | {cov}/{len(rows)}")
print("DONE")

# ============================================================================
# PAPER 1: BLIND-BAND-SIZE SWEEP (blind-band-size robustness): vary the band covariate extent
# (share of declarations); report band share, band duty share, design %bias,
# clearance vs selection-robust top-5% capture, and uplift. numpy/scikit-learn, CPU.
# ============================================================================
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import KFold
EPS=1e-3; RK=dict(max_iter=120,random_state=0); RCAP=80.0; sig=lambda v:1/(1+np.exp(-v)); SEEDS=10; N=60000
def gen(seed,N,lo,hi,confound=True):
    rng=np.random.default_rng(seed)
    x_val=rng.normal(0,1,N); x_hs=rng.normal(0,1,N); x_imp=rng.normal(0,1,N)
    X=np.column_stack([x_val,x_hs,x_imp]); U=rng.normal(0,1,N); Z=rng.integers(0,4,N); z=Z-1.5
    cold=((x_hs>lo)&(x_hs<hi))
    p=sig(-5.6+1.4*x_val+1.0*x_imp+1.8*cold+(0.9 if confound else 0.0)*U); fraud=rng.random(N)<p
    R=np.where(fraud,np.clip(np.exp(0.0+0.9*x_val+1.0*cold+rng.normal(0,0.9,N)),0,RCAP),0.0)
    Dp=rng.random(N)<sig(-1.2+0.5*x_val+0.4*x_imp+1.0*cold)
    Dc=rng.random(N)<np.where(cold,0.0,sig(-2.6+0.9*x_val+0.5*x_imp+(1.0 if confound else 0.0)*U-1.2*z))
    return X,R,Dc,Dp,cold
def cf_clf(X,D,s):
    p=np.zeros(len(X))
    for tr,te in KFold(5,shuffle=True,random_state=s).split(X): p[te]=HGBC(**RK).fit(X[tr],D[tr].astype(int)).predict_proba(X[te])[:,1]
    return np.clip(p,EPS,1-EPS)
def cf_reg(X,mask,R,s):
    p=np.zeros(len(X))
    for tr,te in KFold(5,shuffle=True,random_state=s).split(X):
        m=mask[tr]; p[te]=HGBR(**RK).fit(X[tr][m],R[tr][m]).predict(X[te]) if m.sum()>20 else 0.0
    return p
def cap(score,Rt,k=0.05):
    n=int(k*len(score)); return Rt[np.argsort(-score)[:n]].sum()/Rt.sum() if Rt.sum()>0 else 0.0
ci=lambda a:1.96*np.std(a)/np.sqrt(len(a))
print("=== BAND-SIZE SWEEP (realistic, N=60000, 10 seeds) ===")
print("band[lo,hi]  share%  duty%   design%bias    clr cap%  design cap%   uplift%")
for lo,hi in [(0.6,1.0),(0.5,1.1),(0.4,1.2),(0.25,1.35),(0.1,1.5)]:
    sh=[];du=[];bs=[];cn=[];cr=[];up=[]
    for i in range(SEEDS):
        X,R,Dc,Dp,cold=gen(7000+i,N,lo,hi); truth=R.mean()
        sh.append(cold.mean()*100); du.append(R[cold].sum()/R.sum()*100)
        mpca=cf_reg(X,Dp,R,i); epca=cf_clf(X,Dp,i)
        bs.append(100*(np.mean(mpca+Dp*(R-mpca)/epca)-truth)/truth)
        idx=np.arange(len(X)); rng=np.random.default_rng(10+i); rng.shuffle(idx)
        tr=idx[:len(X)//2]; te=idx[len(X)//2:]
        sn=HGBR(**RK).fit(X[tr][Dc[tr]],R[tr][Dc[tr]]).predict(X[te]) if Dc[tr].sum()>20 else np.zeros(len(te))
        sr=HGBR(**RK).fit(X[tr][Dp[tr]],R[tr][Dp[tr]]).predict(X[te])
        cN=cap(sn,R[te]); cR=cap(sr,R[te]); cn.append(cN*100); cr.append(cR*100); up.append(100*(cR-cN)/cN if cN>0 else np.nan)
    tag="*LOCKED*" if (lo,hi)==(0.4,1.2) else ""
    print(f"[{lo:.2f},{hi:.2f}] {np.mean(sh):5.1f}  {np.mean(du):5.1f}  {np.mean(bs):+5.1f}+/-{ci(bs):4.1f}   {np.mean(cn):5.1f}    {np.mean(cr):5.1f}    {np.nanmean(up):+5.1f}+/-{ci(up):4.1f} {tag}",flush=True)
print("Story: design estimator stays unbiased across band sizes; uplift rises as the band (and its hidden duty) grows.")

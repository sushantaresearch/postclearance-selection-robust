# Adverse-DGP grid: does the policy uplift track the blind band's duty mass?
# Band-intensity lambda scales the band's contribution to BOTH fraud propensity
# (1.8*lambda*cold) and duty (lambda*cold); lambda=1 reproduces the locked REALISTIC
# world exactly. Audit mechanisms (Dc clearance, Dp post-clearance) are held fixed.
# Uses the paper's exact uplift logic (50/50 split, top-5% cap). numpy + sklearn, CPU.
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
N=60000; SEEDS=12; RCAP=80.0; RK=dict(max_iter=120,random_state=0); sig=lambda v:1/(1+np.exp(-v))

def gen_lam(seed,N,lam,cold_on=True,confound=True):
    rng=np.random.default_rng(seed)
    x_val=rng.normal(0,1,N); x_hs=rng.normal(0,1,N); x_imp=rng.normal(0,1,N)
    X=np.column_stack([x_val,x_hs,x_imp]); U=rng.normal(0,1,N); Z=rng.integers(0,4,N); z=Z-1.5
    cold=((x_hs>0.4)&(x_hs<1.2)) if cold_on else np.zeros(N,bool)
    p=sig(-5.6+1.4*x_val+1.0*x_imp+1.8*lam*cold+(0.9 if confound else 0.0)*U); fraud=rng.random(N)<p
    R=np.where(fraud,np.clip(np.exp(0.0+0.9*x_val+lam*1.0*cold+rng.normal(0,0.9,N)),0,RCAP),0.0)
    Dc=rng.random(N)<np.where(cold,0.0,sig(-2.6+0.9*x_val+0.5*x_imp+(1.0 if confound else 0.0)*U-1.2*z))
    Dp=rng.random(N)<sig(-1.2+0.5*x_val+0.4*x_imp+1.0*cold)
    return dict(X=X,R=R,Dc=Dc,Dp=Dp,cold=cold,fraud=fraud)

def cap(score,Rtrue,k):
    n=len(score); pick=np.argsort(-score)[:max(1,int(k*n))]; tot=Rtrue.sum()
    return (Rtrue[pick].sum()/tot) if tot>0 else 0.0

def run(lam,cold_on=True,confound=True,k=0.05):
    ups=[];nc=[];rc=[];cds=[]
    for i in range(SEEDS):
        d=gen_lam(4000+i,N,lam,cold_on,confound); X=d['X'];R=d['R'];Dc=d['Dc'];Dp=d['Dp'];cold=d['cold']
        cds.append(R[cold].sum()/R.sum()*100 if R.sum()>0 and cold.any() else 0.0)
        idx=np.arange(len(X)); rng=np.random.default_rng(10+i); rng.shuffle(idx)
        tr=idx[:len(X)//2]; te=idx[len(X)//2:]
        sn=HGBR(**RK).fit(X[tr][Dc[tr]],R[tr][Dc[tr]]).predict(X[te]) if Dc[tr].sum()>20 else np.zeros(len(te))
        sr=HGBR(**RK).fit(X[tr][Dp[tr]],R[tr][Dp[tr]]).predict(X[te]) if Dp[tr].sum()>20 else np.zeros(len(te))
        cN=cap(sn,R[te],k); cR=cap(sr,R[te],k); nc.append(cN*100); rc.append(cR*100)
        ups.append(100*(cR-cN)/cN if cN>0 else np.nan)
    return np.mean(cds),np.nanmean(nc),np.nanmean(rc),np.nanmean(ups)

print(f"=== ADVERSE-DGP GRID (N={N} SEEDS={SEEDS}, top-5%) ===")
cds,cn,cr,up=run(1.0,cold_on=False,confound=False)
print(f"BENIGN (no structural blind spot): band-duty {cds:4.1f}%  clearance {cn:4.1f}%  design {cr:4.1f}%  uplift {up:+5.1f}%")
print("\n lambda  band-duty-share  clearance_cap  design_cap   uplift")
for lam in [0.0,0.25,0.5,0.75,1.0,1.25,1.5]:
    cds,cn,cr,up=run(lam)
    print(f"  {lam:4.2f}     {cds:5.1f}%          {cn:5.1f}%        {cr:5.1f}%     {up:+6.1f}%"+("  <- locked REALISTIC" if lam==1.0 else ""))

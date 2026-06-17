# ============================================================================
# PAPER 1: CALIBRATED BACKBONE (single reproducibility cell): A operating points,
# B estimation bias + identification ladder (Manski / point ID / MSM Gamma*),
# C held-out policy uplift, D inspected-only evaluation bias. numpy/pandas/sklearn, CPU.
# ============================================================================
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import KFold
N=60000; SEEDS=12
EPS=1e-3; RK=dict(max_iter=120,random_state=0); RCAP=80.0
GRID=np.round(np.concatenate([np.arange(1.00,1.51,0.02),np.arange(1.55,3.01,0.05)]),3)
sig=lambda v:1/(1+np.exp(-v))

def gen(seed,N,confound,cold_on):                      # locked calibrated DGP (== ABL/uplift world)
    rng=np.random.default_rng(seed)
    x_val=rng.normal(0,1,N); x_hs=rng.normal(0,1,N); x_imp=rng.normal(0,1,N)
    X=np.column_stack([x_val,x_hs,x_imp]); U=rng.normal(0,1,N); Z=rng.integers(0,4,N); z=Z-1.5
    cold=((x_hs>0.4)&(x_hs<1.2)) if cold_on else np.zeros(N,bool)
    p=sig(-5.6+1.4*x_val+1.0*x_imp+1.8*cold+(0.9 if confound else 0.0)*U); fraud=rng.random(N)<p
    R=np.where(fraud,np.clip(np.exp(0.0+0.9*x_val+1.0*cold+rng.normal(0,0.9,N)),0,RCAP),0.0)
    Dc=rng.random(N)<np.where(cold,0.0,sig(-2.6+0.9*x_val+0.5*x_imp+(1.0 if confound else 0.0)*U-1.2*z))
    Dp=rng.random(N)<sig(-1.2+0.5*x_val+0.4*x_imp+1.0*cold); Db=rng.random(N)<0.20
    return dict(X=X,R=R,Dc=Dc,Dp=Dp,Db=Db,cold=cold,fraud=fraud)

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
        if lo<=ref<=hi: return G
    return None
def cap(score,Rtrue,k):
    n=len(score); pick=np.argsort(-score)[:max(1,int(k*n))]; tot=Rtrue.sum()
    return (Rtrue[pick].sum()/tot) if tot>0 else 0.0
def gini(a):
    a=np.sort(a); n=len(a)
    return (2*np.sum(np.arange(1,n+1)*a)/(n*a.sum())-(n+1)/n) if a.sum()>0 else 0.0

print(f"==== CALIBRATED BACKBONE  N={N} SEEDS={SEEDS} ====")
prev=[];insp=[];hit=[];coldsh=[];gp=[]
for s in range(4):
    d=gen(1000+s,N,True,True); R=d['R'];Dc=d['Dc'];cold=d['cold'];fr=d['fraud']
    prev.append(fr.mean()*100);insp.append(Dc.mean()*100);hit.append(fr[Dc].mean()*100)
    coldsh.append(R[cold].sum()/R.sum()*100);gp.append(gini(R[R>0]))
print(f"A. prevalence {np.mean(prev):.1f}%  inspection {np.mean(insp):.1f}%  hit {np.mean(hit):.1f}%  Gini|R>0 {np.mean(gp):.3f}  cold-duty-share {np.mean(coldsh):.0f}%")

print("B. ESTIMATION BIAS + IDENTIFICATION LADDER (% vs true E[R])")
def biasrow(tag,cf,cold_on):
    aN=[];aI=[];aD=[];aP=[];aA2=[];aA=[];gso=[];cov=0;mlo=[];mhi=[];mw=[]
    for i in range(SEEDS):
        d=gen(3000+i,N,cf,cold_on); X=d['X'];R=d['R'];Dc=d['Dc'];Dp=d['Dp'];Db=d['Db']
        truth=R.mean(); anchor=R[Db].mean()
        e=cf_clf(X,Dc,i); m=cf_reg(X,Dc,R,i); epca=cf_clf(X,Dp,i); mpca=cf_reg(X,Dp,R,i)
        eo=e[Dc]; yo=R[Dc]
        ipw=np.sum(yo/eo)/np.sum(1/eo); dr=np.mean(m+Dc*(R-m)/e)
        plug=np.where(Dc,R,mpca).mean(); aipw=np.mean(mpca+Dp*(R-mpca)/epca)
        obs=Dc|Dp; lo=(R*obs).sum()/len(R); hi=(R*obs+(~obs)*RCAP).sum()/len(R)
        pb=lambda v:100*(v-truth)/truth
        aN.append(pb(yo.mean()));aI.append(pb(ipw));aD.append(pb(dr));aP.append(pb(plug));aA2.append(pb(aipw));aA.append(pb(anchor))
        gso.append(gstar(yo,eo,truth)); l2,h2=msm(yo,eo,2.0); cov+=(l2<=truth<=h2)
        mlo.append(pb(lo));mhi.append(pb(hi));mw.append(100*(hi-lo)/truth)
    f=lambda a:f"{np.mean(a):+6.1f}+/-{1.96*np.std(a)/np.sqrt(SEEDS):.1f}"
    go=[x for x in gso if x]; rg=f"{np.median(go):.2f}[{min(go):.2f}-{max(go):.2f}]" if go else ">grid"
    print(f"  [{tag:13s}] naive {f(aN)}  IPW_clr {f(aI)}  AIPW_clr {f(aD)}")
    print(f"  {'':15s} delayed-plugin {f(aP)}  PCA-AIPW(design) {f(aA2)}  blind-anchor {f(aA)}")
    print(f"  {'':15s} MSM Gamma*_oracle {rg}  covers truth @G=2.0 {cov}/{SEEDS}")
    print(f"  {'':15s} Manski floor {np.mean(mlo):+.0f}% ceiling {np.mean(mhi):+.0f}% (ceiling vacuous; covers {SEEDS}/{SEEDS})")
biasrow("BENIGN",False,False); biasrow("CONFOUND-ONLY",True,False); biasrow("REALISTIC",True,True)

print("C. HELD-OUT POLICY UPLIFT (50/50 disjoint, top-5%)")
def uplift(tag,cf,cold_on,k=0.05):
    ups=[];nc=[];rc=[]
    for i in range(SEEDS):
        d=gen(4000+i,N,cf,cold_on); X=d['X'];R=d['R'];Dc=d['Dc'];Dp=d['Dp']
        idx=np.arange(len(X)); rng=np.random.default_rng(10+i); rng.shuffle(idx)
        tr=idx[:len(X)//2]; te=idx[len(X)//2:]
        sn=HGBR(**RK).fit(X[tr][Dc[tr]],R[tr][Dc[tr]]).predict(X[te]) if Dc[tr].sum()>20 else np.zeros(len(te))
        sr=HGBR(**RK).fit(X[tr][Dp[tr]],R[tr][Dp[tr]]).predict(X[te]) if Dp[tr].sum()>20 else np.zeros(len(te))
        cN=cap(sn,R[te],k); cR=cap(sr,R[te],k); nc.append(cN*100); rc.append(cR*100)
        ups.append(100*(cR-cN)/cN if cN>0 else np.nan)
    print(f"  [{tag:10s}] clearance cap@5% {np.nanmean(nc):4.1f}%   PCA-design cap {np.nanmean(rc):4.1f}%   uplift {np.nanmean(ups):+5.1f}+/-{1.96*np.nanstd(ups)/np.sqrt(SEEDS):.1f}")
uplift("BENIGN",False,False); uplift("REALISTIC",True,True)

infN=[];infR=[]
for i in range(SEEDS):
    d=gen(5000+i,N,True,True); X=d['X'];R=d['R'];Dc=d['Dc'];Dp=d['Dp']
    idx=np.arange(len(X)); rng=np.random.default_rng(20+i); rng.shuffle(idx)
    tr=idx[:len(X)//2]; te=idx[len(X)//2:]
    sn=HGBR(**RK).fit(X[tr][Dc[tr]],R[tr][Dc[tr]]).predict(X[te]); sr=HGBR(**RK).fit(X[tr][Dp[tr]],R[tr][Dp[tr]]).predict(X[te])
    Rte=R[te]; ins=Dc[te]
    infN.append((cap(sn[ins],Rte[ins],.05)-cap(sn,Rte,.05))*100); infR.append((cap(sr[ins],Rte[ins],.05)-cap(sr,Rte,.05))*100)
fm=lambda a:f"{np.mean(a):+.1f}+/-{1.96*np.std(a)/np.sqrt(SEEDS):.1f}"
print(f"D. INSPECTED-ONLY CAPTURE-EVAL BIAS (FIELD-minus-ORACLE, top-5%, REALISTIC): naive {fm(infN)} pts  robust {fm(infR)} pts")
print("DONE")

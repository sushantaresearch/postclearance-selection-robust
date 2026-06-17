# Proposition 6 verification: minimum audit-allocation rule for the blind band.
# Tests that b* = max(b_prec, b_conf) predicts the empirical operating point on the
# paper's locked DGP. Self-contained; CPU; numpy + sklearn HistGradientBoosting.
# Pin: numpy and scikit-learn versions should match the released environment.
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REF_N=300000; MBAR_ITERS=400; POOL_N=600000; N_POP=60000; S_VAR=300; REPS_CONF=100
alpha=0.10; s_target=0.15; delta=0.05   # targets used in PART 3

def gen(seed,N,confound=True,cold_on=True,cold_pca=1.0,cold_mult=1.0,prev=-5.6,confU=1.0):
    rng=np.random.default_rng(seed)
    x_val=rng.normal(0,1,N); x_hs=rng.normal(0,1,N); x_imp=rng.normal(0,1,N)
    U=rng.normal(0,1,N); Z=rng.integers(0,4,N)
    cold=((x_hs>0.4)&(x_hs<1.2)).astype(float) if cold_on else np.zeros(N)
    sig=lambda t:1/(1+np.exp(-t))
    p=sig(prev+1.4*x_val+1.0*x_imp+1.8*cold+(0.9 if confound else 0.0)*U)
    fraud=(rng.random(N)<p).astype(float)
    R=np.where(fraud>0,np.clip(np.exp(0.0+0.9*x_val+cold_mult*cold+rng.normal(0,0.9,N)),0,80),0.0)
    return np.column_stack([x_val,x_hs,x_imp,Z.astype(float)]),R,cold

# true outcome model m(x)=E[R|x] (large reference sample)
Xr,Rr,_=gen(1,REF_N)
mbar=HistGradientBoostingRegressor(max_depth=3,learning_rate=0.07,max_iter=MBAR_ITERS,
                                   min_samples_leaf=80).fit(Xr,Rr)

# blind-band pool; tau2, omega2 from the same pool used for sampling
Xp,Rp,coldp=gen(7,POOL_N); bmask=coldp==1
mb_pool=mbar.predict(Xp[bmask]); Rb_pool=Rp[bmask]
pi_B=bmask.mean(); tau2=np.var(mb_pool,ddof=0); omega2=np.mean((Rb_pool-mb_pool)**2)
n_B=int(round(N_POP*pi_B)); P=len(Rb_pool); thetaB=Rb_pool.mean()
print(f"pi_B={pi_B:.3f}  n_B={n_B}  tau2={tau2:.3f}  omega2={omega2:.3f}  poolband={P}")
Rb=Rb_pool[:n_B].copy(); mb=mb_pool[:n_B].copy()   # fixed band sample for conformal

b_grid=np.array([0.01,0.02,0.03,0.05,0.08,0.12,0.20,0.35,0.60,1.00])
conf_grid=b_grid[b_grid<1.0]   # drop full-audit (no held-out test set)

print("\nPART 1  variance scaling  sigma_B^2(b)=tau2+omega2/b   SE=sqrt(sigma_B^2/n_B)")
print(" b      emp_SE   pred_SE   ratio    bias")
for b in b_grid:
    est=np.empty(S_VAR)
    for s in range(S_VAR):
        r=np.random.default_rng(1000+s+int(b*1e6))
        idx=r.integers(0,P,n_B); Dp=(r.random(n_B)<b).astype(float)
        est[s]=np.mean(mb_pool[idx]+Dp*(Rb_pool[idx]-mb_pool[idx])/b)
    emp=est.std(ddof=1); pred=np.sqrt((tau2+omega2/b)/n_B)
    print(f" {b:5.2f}  {emp:7.4f}  {pred:7.4f}  {emp/pred:6.3f}  {est.mean()-thetaB:+.4f}")

print(f"\nPART 2  split-conformal on band (alpha={alpha}, nominal {1-alpha:.2f})")
print(" b     n_cal  finite  emp_cov  overcov  1/(nc+1)  medwidth")
for b in conf_grid:
    cov=[]; wid=[]; nc=[]
    for s in range(REPS_CONF):
        r=np.random.default_rng(5000+s+int(b*1e6)); Dp=(r.random(n_B)<b)
        cal=np.where(Dp)[0]; te=np.where(~Dp)[0]; ncal=len(cal)
        if ncal<1 or len(te)<1: continue
        nc.append(ncal); k=int(np.ceil((1-alpha)*(ncal+1)))
        if k>ncal: cov.append(1.0); wid.append(np.inf); continue
        q=np.sort(np.abs(Rb[cal]-mb[cal]))[k-1]
        cov.append(np.mean((Rb[te]>=mb[te]-q)&(Rb[te]<=mb[te]+q))); wid.append(2*q)
    ncm=np.mean(nc); fin=int(np.ceil((1-alpha)*(ncm+1)))<=ncm
    fw=[w for w in wid if np.isfinite(w)]
    print(f" {b:5.2f}  {ncm:5.0f}  {str(fin):5s}  {np.mean(cov):6.3f}  {np.mean(cov)-(1-alpha):+6.3f}  {1/(ncm+1):7.4f}  {(np.median(fw) if fw else float('inf')):7.3f}")

print(f"\nPART 3  composite b* = max(b_prec,b_conf)   s_target={s_target}, delta={delta}")
s_floor=np.sqrt(tau2/n_B)
b_prec=omega2/(n_B*s_target**2-tau2) if n_B*s_target**2>tau2 else float('inf')
b_conf=max((1-alpha)/alpha,1/delta-1)/n_B
b_star=max(b_prec,b_conf)
print(f"  precision floor sqrt(tau2/n_B)={s_floor:.4f}  b_prec={b_prec:.4f}  b_conf={b_conf:.4f}  b*={b_star:.4f}  pi_B*b*={pi_B*b_star:.4f}")
for b in sorted(set([max(0.005,round(0.5*b_star,4)),round(b_star,4),round(1.5*b_star,4),round(2*b_star,4)])):
    est=np.empty(200)
    for s in range(200):
        r=np.random.default_rng(9000+s+int(b*1e6)); idx=r.integers(0,P,n_B); Dp=(r.random(n_B)<b).astype(float)
        est[s]=np.mean(mb_pool[idx]+Dp*(Rb_pool[idx]-mb_pool[idx])/b)
    se=est.std(ddof=1); cov=[]
    for s in range(REPS_CONF):
        r=np.random.default_rng(11000+s+int(b*1e6)); Dp=(r.random(n_B)<b)
        cal=np.where(Dp)[0]; te=np.where(~Dp)[0]; ncal=len(cal)
        if ncal<1 or len(te)<1: continue
        k=int(np.ceil((1-alpha)*(ncal+1)))
        if k>ncal: cov.append(1.0); continue
        q=np.sort(np.abs(Rb[cal]-mb[cal]))[k-1]; cov.append(np.mean((Rb[te]>=mb[te]-q)&(Rb[te]<=mb[te]+q)))
    oc=np.mean(cov)-(1-alpha)
    print(f"   b={b:.4f}: SE={se:.4f} ({'OK' if se<=s_target else 'FAIL'})  overcov={oc:+.3f} ({'OK' if oc<=delta else 'FAIL'})  both={se<=s_target and oc<=delta}")
print("Expected: both OK for b>=b*, at least one FAIL below b*.")

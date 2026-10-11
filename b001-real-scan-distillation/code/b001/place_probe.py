import sys, json, numpy as np, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
B='https://vesuvius-challenge-open-data.s3.amazonaws.com/'
def fetch(u):
    for t in range(5):
        try: return urllib.request.urlopen(u,timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code in (403,404): return None
            raise
        except Exception:
            time.sleep(2**t)
    raise RuntimeError(u)
def zcrop(root, level, y0, x0, h, w):
    m=json.loads(fetch(f'{B}{root}/{level}/.zarray')); nz,cy,cx=m['chunks']
    out=np.zeros((m['shape'][0],h,w),np.uint8)
    jobs=[(iy,ix) for iy in range(y0//cy,(y0+h-1)//cy+1) for ix in range(x0//cx,(x0+w-1)//cx+1)]
    def get(j):
        iy,ix=j; raw=fetch(f'{B}{root}/{level}/0/{iy}/{ix}')
        return j, None if raw is None else np.frombuffer(raw,np.uint8).reshape(nz,cy,cx)
    with ThreadPoolExecutor(16) as ex:
        for (iy,ix),c in ex.map(get,jobs):
            if c is None: continue
            ys,xs=iy*cy,ix*cx
            a0,a1=max(y0,ys),min(y0+h,ys+cy); b0,b1=max(x0,xs),min(x0+w,xs+cx)
            out[:,a0-y0:a1-y0,b0-x0:b1-x0]=c[:,a0-ys:a1-ys,b0-xs:b1-xs]
    return out, m['shape']
def ncc(a,b,m):
    a=a[m]; b=b[m]
    if a.size<1000: return np.nan
    a=a-a.mean(); b=b-b.mean(); d=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d>0 else np.nan
def probe(lr_root, hr_root, k, ncrop=4, S=192, R=16, DZ=3):
    import cv2
    m=json.loads(fetch(f'{B}{lr_root}/0/.zarray')); L,H,W=m['shape']
    ov,_=zcrop(lr_root,4,0,0,H//16,W//16)  # coverage at level 4
    cov=(ov>0).any(0).astype(np.float32)
    # candidate crop centres on a grid, keep fully covered, pick spread
    g=S//16; cands=[]
    for yy in range(0,cov.shape[0]-g,g):
        for xx in range(0,cov.shape[1]-g,g):
            if cov[yy:yy+g,xx:xx+g].mean()>0.98: cands.append((yy*16,xx*16))
    rng=np.random.default_rng(0); 
    if not cands: return {'error':'no covered crop'}
    pick=[cands[i] for i in rng.choice(len(cands),min(ncrop,len(cands)),replace=False)]
    hm=json.loads(fetch(f'{B}{hr_root}/2/.zarray')); hL=hm['shape'][0]
    res=[]
    for (y0,x0) in pick:
        lr,_=zcrop(lr_root,0,y0-R,x0-R,S+2*R,S+2*R); lr=lr.astype(np.float32)
        # HR level 2: in-plane px = 4 HR voxels; LR px = k HR voxels -> HR L2 coords = LR coords * k/4
        f=k/4.0
        hy0,hx0=int(np.floor(y0*f))-2,int(np.floor(x0*f))-2; hh,hw=int(np.ceil(S*f))+4,int(np.ceil(S*f))+4
        hr,_=zcrop(hr_root,2,hy0,hx0,hh,hw); hr=hr.astype(np.float32)
        # pool layers hL -> L
        edges=np.linspace(0,hL,L+1)
        hp=np.stack([hr[int(round(edges[i])):int(round(edges[i+1]))].mean(0) for i in range(L)])
        # resample in-plane onto LR grid of the crop (y0..y0+S)
        ys=(np.arange(S)+0.5)*f+ (y0*f-hy0) -0.5; xs=(np.arange(S)+0.5)*f+(x0*f-hx0)-0.5
        mx,my=np.meshgrid(xs,ys)
        hq=np.stack([cv2.remap(hp[i],mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR) for i in range(L)])
        valid=(hq>0).all(0)
        best=(-9,0,0,0); n0=None; cz=L//2; zs=slice(cz-6,cz+7)
        for dz in range(-DZ,DZ+1):
            for dy in range(-R,R+1):
                for dx in range(-R,R+1):
                    a=lr[cz-6+dz:cz+7+dz, R+dy:R+dy+S, R+dx:R+dx+S]
                    v=valid&(a>0).all(0)
                    c=ncc(a,hq[zs],np.broadcast_to(v,a.shape))
                    if dz==0 and dy==0 and dx==0: n0=c
                    if c>best[0]: best=(c,dz,dy,dx)
        res.append({'crop':[y0,x0],'ncc0':round(n0,3),'best':round(best[0],3),'dz':best[1],'dy':best[2],'dx':best[3]})
        print(res[-1],flush=True)
    return res
if __name__=='__main__':
    lr,hr,k=sys.argv[1],sys.argv[2],float(sys.argv[3])
    print(json.dumps(probe(lr,hr,k)))

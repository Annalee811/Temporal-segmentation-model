import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

def topology(mask):
    sk=skeletonize(mask);p=np.pad(sk,1);h,w=sk.shape;degree=np.zeros(sk.shape,np.uint8)
    for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
        edge=p[1+dy:1+dy+h,1+dx:1+dx+w].copy()
        if dy and dx:edge&=~(p[1+dy:1+dy+h,1:1+w]|p[1:1+h,1+dx:1+dx+w])
        degree+=edge.astype(np.uint8)
    return sk,sk&(degree==1),sk&(degree>=3)

def proximity(points,sigma=10):
    return np.exp(-ndi.distance_transform_edt(~points)**2/(2*sigma*sigma)).astype(np.float32) if points.any() else np.zeros(points.shape,np.float32)

def orientation(sk,sigma=4):
    s=sk.astype(np.float32);w=ndi.gaussian_filter(s,sigma);den=np.maximum(w,1e-6)
    mx=ndi.gaussian_filter(s,sigma,order=(0,1))*sigma**2/den;my=ndi.gaussian_filter(s,sigma,order=(1,0))*sigma**2/den
    xx=(ndi.gaussian_filter(s,sigma,order=(0,2))*sigma**4+sigma**2*w)/den-mx*mx
    yy=(ndi.gaussian_filter(s,sigma,order=(2,0))*sigma**4+sigma**2*w)/den-my*my
    xy=ndi.gaussian_filter(s,sigma,order=(1,1))*sigma**4/den-mx*my
    norm=np.sqrt((xx-yy)**2+4*xy*xy);coherence=np.clip(norm/np.maximum(xx+yy,1e-6),0,1)
    return (xx-yy)/np.maximum(norm,1e-6),2*xy/np.maximum(norm,1e-6),coherence

def extra_features(previous,current):
    pm=previous.astype(np.float32);cm=current.astype(np.float32);ps,pt,pj=topology(previous);cs,ct,cj=topology(current)
    po=[ndi.gaussian_filter(pm,s) for s in [3,8,16]];co=ndi.gaussian_filter(cm,8)
    overlap=ndi.gaussian_filter(pm*cm,8)/np.maximum(co,1e-6)
    unmatched=ndi.gaussian_filter(cm*(~ndi.binary_dilation(previous,iterations=2)),8)/np.maximum(co,1e-6)
    ca,cb,cc=orientation(cs);pa,pb,pc=orientation(ps)
    if ps.any():
        dist,idx=ndi.distance_transform_edt(~ps,return_indices=True);align=np.clip((1+ca*pa[tuple(idx)]+cb*pb[tuple(idx)])*.5,0,1);pco=pc[tuple(idx)]
    else:dist=np.full(pm.shape,100,np.float32);align=np.zeros_like(pm);pco=np.zeros_like(pm)
    score=align*np.exp(-dist*dist/128)*np.sqrt(cc*pco)
    ratio=np.minimum(po[1],co)/np.maximum(np.maximum(po[1],co),1e-6)
    result=np.stack([*po,overlap,unmatched,proximity(ct),proximity(cj),proximity(pt),proximity(pj),align,pco,cc,score,ratio],2)
    assert result.shape[2]==14 and np.isfinite(result).all()
    return result.astype(np.float32)

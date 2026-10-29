import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

def graph(mask):
    sk=skeletonize(mask);nodes=np.argwhere(sk);lookup={tuple(p):i for i,p in enumerate(nodes)};adj=[[] for _ in nodes]
    for i,(y,x) in enumerate(nodes):
        for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            j=lookup.get((y+dy,x+dx))
            if j is None:continue
            if dy and dx and ((y+dy,x) in lookup or (y,x+dx) in lookup):continue
            adj[i].append(j)
    return sk,nodes,adj

def chains(mask):
    sk,nodes,adj=graph(mask);seen=set();paths=[]
    starts=[i for i,a in enumerate(adj) if len(a)!=2]+list(range(len(nodes)))
    for start in starts:
        for nxt in adj[start]:
            edge=tuple(sorted((start,nxt)))
            if edge in seen:continue
            seen.add(edge);path=[start,nxt];prev,cur=start,nxt
            while len(adj[cur])==2:
                nxt=next(v for v in adj[cur] if v!=prev);edge=tuple(sorted((cur,nxt)))
                if edge in seen:break
                seen.add(edge);path.append(nxt);prev,cur=cur,nxt

            if len(adj[path[0]])>=3:path=path[1:]
            if path and len(adj[path[-1]])>=3:path=path[:-1]
            if len(path)>=4 and path[0]!=path[-1]:paths.append(np.asarray(path,dtype=np.int64))
    return sk,nodes,paths,adj

def path_loss(logits,gt):
    """Train only same-class human skeleton runs; never smooth true class changes."""
    import torch
    terms=[];prob=logits.softmax(1)
    for b,yy in enumerate(gt.detach().cpu().numpy()):
        _,nodes,paths,_=chains((yy>0)&(yy<255));used=0
        for path in paths:
            xy=nodes[path];values=yy[tuple(xy.T)];bounds=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1,len(values)]
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                if hi-lo<8:continue
                ix=xy[lo:hi:2];p=prob[b,:,ix[:,0],ix[:,1]];label=int(values[lo])-1
                terms.append(-torch.log(p[label].mean().clamp_min(1e-6))+.5*(p-p.mean(1,keepdim=True)).square().sum(0).mean());used+=1
                if used>=32:break
            if used>=32:break
    return torch.stack(terms).mean() if terms else logits.sum()*0

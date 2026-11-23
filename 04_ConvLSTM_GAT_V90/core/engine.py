from paths import ROOT,WORK
"""Whole-segment continuity decoding across learned junction pairings; fixed root geometry."""

import argparse,json,pickle

import numpy as np

import torch

from PIL import Image

from scipy import ndimage as ndi

from scipy.spatial import cKDTree

from topology import routes,fuse_crossing_bridges

from train import group_score,PairHead

from paths import ROOT,OUT,LAST,BASE,PERIODS,js,csvout,sha

def runs(v):
 cut=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1,len(v)]
 return [(int(a),int(b),int(v[a])) for a,b in zip(cut[:-1],cut[1:]) if b>a]

def stable(v,from_start=True):
 v=v if from_start else v[::-1]
 for k in range(min(24,len(v))):
  q=v[k:k+8]
  if len(q)<4:continue
  for c in [1,2]:
   if (q==c).sum()>=max(4,int(np.ceil(.875*len(q)))):return c,float((q==c).mean())
 return 0,0.

def review_setup():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 class Args: prototype=False
 a=Args()
 dest=OUT/'final'
 with (OUT/'data/topology.pkl').open('rb') as f:top=pickle.load(f)
 fusion_records=[]
 for tt in PERIODS:
  fusion_records.extend([dict(period=tt,**q) for q in fuse_crossing_bridges(top['topos'][tt])])

 result=json.loads((OUT/('pair_training_results.json' if a.prototype else 'joint_training_results.json')).read_text());logit=np.load(OUT/('pair_logits.npy' if a.prototype else 'joint_pair_logits.npy'));raw=np.load(OUT/'data/pairs.npz')['raw'];groups={(g['period'],g['hub']):g for g in top['groups']};armrec={(t,j):aa for t,items in top['arms'].items() for j,aa in items};weight=result['geometry_weight'];confidence=result['assignment_confidence'];owners=dict(np.load(LAST/'data/unet/ownership.npz'))
 pairdata=dict(np.load(OUT/'data/pairs.npz'));pairmodel=PairHead(pairdata['x'].shape[1]);pairck=torch.load(OUT/'best_pair_head.pt' if a.prototype else OUT/'joint_best.pt',weights_only=False,map_location='cpu');pairmodel.load_state_dict(pairck['state_dict'] if a.prototype else pairck['pair']);pairmodel.eval();embeddings=np.load(LAST/'inference.npz' if a.prototype else OUT/'joint_inference.npz')['embedding'];graphraw=np.load(ROOT/'raw_graph.npz')['features'];rgb={t:np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')).astype(np.float32)/255 for t in PERIODS}
 cached=dict(np.load(LAST/'inference.npz' if a.prototype else OUT/'joint_inference.npz'));h=cached['h'];ck=torch.load(LAST/'best.pt' if a.prototype else OUT/'joint_best.pt',weights_only=False,map_location='cpu');state=ck['state_dict'] if a.prototype else ck['backbone'];w=state['root_state.weight'].numpy()[0];bias=float(state['root_state.bias'][0]);node_new=1/(1+np.exp(-(h@w+bias)))
 graph=dict(np.load(LAST/'data/unet/graph.npz'));centers=graph['centers'];raw_identity=np.maximum(h@state['identity.0.weight'].numpy().T+state['identity.0.bias'].numpy(),0)+np.minimum(np.exp(np.minimum(h@state['identity.0.weight'].numpy().T+state['identity.0.bias'].numpy(),0))-1,0)
 current={t:np.array(Image.open(LAST/'full'/f'{t}_classes.png')) for t in PERIODS};old={t:np.array(Image.open(BASE/'unet/full'/f'{t}_classes.png')) for t in PERIODS};spatial=np.load(LAST/'data/spatial.npy');trees={};directions={};tree_points={};support_cache={}
 for t in PERIODS:
  pts=[];vv=[]
  for s in top['topos'][t]['segments']:
   p=s['points'];l=p[np.maximum(np.arange(len(p))-5,0)];r=p[np.minimum(np.arange(len(p))+5,len(p)-1)];v=r-l;v=v/np.linalg.norm(v,axis=1).clip(1e-8)[:,None];pts.extend(p);vv.extend(v)
  tree_points[t]=np.array(pts);trees[t]=cKDTree(tree_points[t]);directions[t]=np.array(vv)
 def support(tt,p,v):
  if len(p)==0:return 0.
  dd,nn=trees[tt].query(p,k=4);align=np.abs((directions[tt][nn]*v[:,None]).sum(-1));return float(((dd<=3.5)&(align>=.9)).any(1).mean())
 def history(ti,p):
  if not len(p):return [0.]*4
  lo=p[np.maximum(np.arange(len(p))-5,0)];hi=p[np.minimum(np.arange(len(p))+5,len(p)-1)];v=hi-lo;v=v/np.linalg.norm(v,axis=1).clip(1e-8)[:,None]
  return [support(t,p,v) for t in PERIODS]
 prior_nodes={tt:np.flatnonzero(graph['phase']==j) for j,tt in enumerate(PERIODS)};prior_node_trees={tt:cKDTree(centers[n]) for tt,n in prior_nodes.items()}

 va,vb=graph['pairs'].T;vi=(graph['split']==1)&(graph['phase'][va]!=graph['phase'][vb]);yy=graph['target'][vi];ss=cached['connection'][vi];best=(-1.,.5)
 for th in np.arange(.1,.96,.05):
  pp=ss>=th;tp0=int((pp&(yy==1)).sum());fp0=int((pp&(yy==0)).sum());fn0=int((~pp&(yy==1)).sum());pr0=tp0/max(tp0+fp0,1);rr0=tp0/max(tp0+fn0,1);ff=1.25*pr0*rr0/max(.25*pr0+rr0,1e-9)
  if ff>best[0]:best=(ff,float(th))
 identity_threshold=best[1]
 def track_evidence(ti,p,other_periods=None):
  if (ti==0 and other_periods is None) or len(p)<2:return 0.,0.
  pp=p[::max(1,len(p)//64)];lo=pp[np.maximum(np.arange(len(pp))-2,0)];hi=pp[np.minimum(np.arange(len(pp))+2,len(pp)-1)];direction=hi-lo;direction=direction/np.linalg.norm(direction,axis=1).clip(1e-8)[:,None];best_support=0.;best_affinity=0.;cur=owners[PERIODS[ti]][pp[:,1],pp[:,0]]
  for tj in (range(ti) if other_periods is None else other_periods):
   tt=PERIODS[tj];dd,nn=prior_node_trees[tt].query(pp,k=8);prev=prior_nodes[tt][nn];prevdir=graphraw[prev,4:6];alignment=np.abs((prevdir*direction[:,None]).sum(-1));a0=prev.ravel();b0=np.repeat(cur,8);delta=centers[b0]-centers[a0];distance=np.linalg.norm(delta,axis=1);attrs=np.c_[distance/32,np.abs(delta/32),np.abs((graphraw[a0,4:6]*graphraw[b0,4:6]).sum(1)),np.full(len(a0),abs(ti-tj)),np.zeros(len(a0)),np.abs(graphraw[a0,6]-graphraw[b0,6]),np.minimum(distance,100)/100].astype(np.float32);feat=np.c_[np.abs(raw_identity[a0]-raw_identity[b0]),raw_identity[a0]*raw_identity[b0],attrs];z0=feat@state['connection.0.weight'].numpy().T+state['connection.0.bias'].numpy();z0=np.where(z0>=0,z0,np.expm1(np.clip(z0,-30,0)));z0=z0@state['connection.2.weight'].numpy().T+state['connection.2.bias'].numpy();prob=(1/(1+np.exp(-np.clip(z0[:,0],-30,30)))).reshape(len(pp),8);valid=(dd<=32)&(alignment>=.8)&(prob>=identity_threshold);rank=np.where(valid,prob*np.exp(-dd*dd/800)*alignment**2,-1);chosen=rank.argmax(1);good=valid.any(1)
   if not good.any():continue
   picked=prev[np.arange(len(pp)),chosen];shift=centers[cur[good]]-centers[picked[good]];median=np.median(shift,axis=0);coherent=np.linalg.norm(shift-median,axis=1)<=12;coverage=float(coherent.sum()/len(pp));scores=prob[np.arange(len(pp)),chosen][good][coherent];affinity=float(scores.mean()) if len(scores) else 0.
   if coverage*affinity>best_support*best_affinity:best_support=coverage;best_affinity=affinity
  return best_support,best_affinity
 return locals()

def reconstruct(setup,current):
 top=setup['top']
 groups=setup['groups']
 armrec=setup['armrec']
 weight=setup['weight']
 confidence=setup['confidence']
 owners=setup['owners']
 embeddings=setup['embeddings']
 graphraw=setup['graphraw']
 rgb=setup['rgb']
 trees=setup['trees']
 directions=setup['directions']
 pairmodel=setup['pairmodel']
 pairdata=setup['pairdata']
 logit=setup['logit']
 raw=setup['raw']
 junctions=[];by_period={}
 for ti,t in enumerate(PERIODS):
  im=current[t];tp=top['topos'][t];decisions=[];hub_semantic={};hub_chosen={};arm_colors={};uncertain=0
  for ji,hub in enumerate(tp['hubs']):
   if hub.get('disabled'):continue
   arms=hub['arms'];colors=[];purities=[]
   for aa in arms:
    q=aa['body_points'];c,pr=stable(im[q[:,1],q[:,0]]);colors.append(c);purities.append(pr)
   arm_colors[ji]=colors
   known=[c for c in colors if c in [1,2]]
   if known and len(known)==len(colors) and len(set(known))==1:hub_semantic[ji]=known[0]
   g=groups.get((t,ji));chosen=None;prob=0.
   if hub.get('fused_short_bridge') is not None:
    aa=[]
    for arm in arms:
     q=arm['body_points'];node=np.unique(owners[t][q[:,1],q[:,0]]);node=node[node>=0];e=embeddings[node].mean(0);e/=max(np.linalg.norm(e),1e-8);pv=[]
     for tt in PERIODS:
      dd,nn=trees[tt].query(q,k=4);align=np.abs((directions[tt][nn]*arm['direction']).sum(-1));pv.append(float(((dd<=3.5)&(align>=.9)).any(1).mean()))
     aa.append(dict(e=e,width=float(graphraw[node,6].mean()),color=rgb[t][q[:,1],q[:,0]].mean(0),support=np.array(pv)))
    feats=[];local={}
    for i in range(4):
     for j in range(i+1,4):
      u,v=aa[i],aa[j];f=np.r_[np.abs(u['e']-v['e']),u['e']*v['e'],-arms[i]['direction']@arms[j]['direction'],np.linalg.norm(arms[i]['point']-arms[j]['point'])/16,abs(u['width']-v['width']),np.abs(u['color']-v['color']),u['support']+v['support'],np.abs(u['support']-v['support']),np.eye(4)[ti]].astype(np.float32);local[str(i)+','+str(j)]=len(raw)+len(feats);feats.append(f)
    extra=np.array(feats)
    with torch.no_grad():extra_logits=pairmodel(torch.from_numpy((extra-pairdata['mean'])/pairdata['std'])).numpy()
    logit=np.r_[logit,extra_logits];raw=np.concatenate([raw,extra]);g=dict(pair_indices=local,options=[[(0,1),(2,3)],[(0,2),(1,3)],[(0,3),(1,2)]])
   if g:
    score=group_score(g,logit,raw,weight);p=np.exp(score-score.max());p/=p.sum();ix=int(score.argmax());prob=float(p[ix]);option=g['options'][ix];valid=all(-arms[a0]['direction']@arms[b0]['direction']>=.35 for a0,b0 in option)
    geometry_scores=np.array([sum(-arms[u]['direction']@arms[v]['direction'] for u,v in op) for op in g['options']]);geo_best=int(geometry_scores.argmax());ordered=np.sort(geometry_scores);geo_opt=g['options'][geo_best];geo_min=min(-arms[u]['direction']@arms[v]['direction'] for u,v in geo_opt);geo_prob=[1/(1+np.exp(-np.clip(logit[g['pair_indices'][str(u)+','+str(v)]],-30,30))) for u,v in geo_opt];unique_geometry=geo_min>=.82 and ordered[-1]-ordered[-2]>=.35 and min(geo_prob)>=.35
    if unique_geometry and not (prob>=confidence and valid):option=geo_opt;prob=float(p[geo_best]);valid=True
    pair_prob=[1/(1+np.exp(-np.clip(logit[g['pair_indices'][str(u)+','+str(v)]],-30,30))) for u,v in option]
    if ((prob>=confidence and valid) or unique_geometry) and min(pair_prob)>=.35:decisions.append((ji,option,prob));chosen=option;hub_chosen[ji]=(option,prob)
   if chosen is None:uncertain+=1
   junctions.append(dict(period=t,hub=ji,x=float(hub['center'][0]),y=float(hub['center'][1]),arms=len(arms),pairing_accepted=chosen is not None,pairing_confidence=prob,arm_anchor_colors=','.join(map(str,colors)),semantic_consensus=hub_semantic.get(ji,0)))
  paths,rootof,accepted=routes(tp,decisions)
  by_period[t]=dict(paths=paths,rootof=rootof,accepted=accepted,hub_chosen=hub_chosen,arm_colors=arm_colors)
 return by_period,junctions

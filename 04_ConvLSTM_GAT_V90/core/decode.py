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

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 ap=argparse.ArgumentParser();ap.add_argument('--prototype',action='store_true');a=ap.parse_args();dest=OUT/('prototype' if a.prototype else 'final');dest.mkdir(exist_ok=True);(dest/'full').mkdir(exist_ok=True);(dest/'changes').mkdir(exist_ok=True)
 with (OUT/'data/topology.pkl').open('rb') as f:top=pickle.load(f)
 fusion_records=[]
 for tt in PERIODS:
  fusion_records.extend([dict(period=tt,**q) for q in fuse_crossing_bridges(top['topos'][tt])])
 csvout(dest/'Fused_crossing_bridges.csv',fusion_records)
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
 rejected=[];edits=[];junctions=[];paths_out=[];audits=[];remaining=[]
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
  paths,rootof,accepted=routes(tp,decisions);seed=np.zeros(im.shape,np.uint8);reason_seed=np.zeros(im.shape,np.uint8);allowed=np.zeros(im.shape,bool);pindex=0
  if ti:
   for pi,path in enumerate(paths):
    points=[]
    for seg,end in path['parts']:
     q=tp['segments'][seg]['points'];points.extend(q if end==0 else q[::-1])
    p=np.array(points);labels=im[p[:,1],p[:,0]];prior=history(ti,p);past=max(prior[:ti]) if ti else 0.;later=max(prior[ti+1:]) if ti<3 else 0.;nodes=np.unique(owners[t][p[:,1],p[:,0]]);nodes=nodes[nodes>=0];newprob=float(node_new[nodes].mean()) if len(nodes) else 0.;nonext=labels[labels!=3];green=int((labels==2).sum());purple=int((labels==1).sum());greenness=green/max(len(nonext),1)
    newaxis=green>=24 and purple<=max(4,int(.05*green)) and past<.25 and (newprob>=.6 or (greenness>=.95 and past<.1))
    endpoints=[]
    for seg,end in [path['parts'][0],(path['parts'][-1][0],1-path['parts'][-1][1])]:endpoints.append(tp['segments'][seg]['end_hubs'][end])
    changed=0

    anchors=[(a0,b0) for a0,b0,c0 in runs((labels==1).astype(np.int32)) if c0 and b0-a0>=8]
    for (left0,left1),(right0,right1) in zip(anchors[:-1],anchors[1:]):
     lo,hi=left1,right0
     if hi<=lo or not np.isin(labels[lo:hi],[2,3]).any():continue
     coverage,affinity=track_evidence(ti,p[lo:hi]);lc,la=track_evidence(ti,p[max(left0,left1-24):left1]);rc,ra=track_evidence(ti,p[right0:min(right1,right0+24)]);fc,fa=track_evidence(ti,p[lo:hi],list(range(ti+1,4))) if ti<3 else (0.,0.)
     direct=coverage>=.55 and affinity>=.55

     occluded=min(lc,rc)>=.5 and min(la,ra)>=.55 and fc>=.5 and fa>=.55
     if direct or occluded:
      q=p[lo:hi];seed[q[:,1],q[:,0]]=1;reason_seed[q[:,1],q[:,0]]=1;changed+=len(q);edits.append(dict(period=t,predicted_axis=pi,from_class=-1,to_class=1,skeleton_pixels=len(q),x_start=int(q[0,0]),y_start=int(q[0,1]),x_end=int(q[-1,0]),y_end=int(q[-1,1]),past_aligned_support=coverage,later_aligned_support=later,learned_new_probability=newprob,reason=('complete mixed-color bridge with earlier direct same-root support' if direct else 'occluded earlier interior: both flanks existed earlier and later same-root continuation confirms the full bridge')+'; affinities '+str([round(affinity,3),round(la,3),round(ra,3),round(fa,3)])))
     else:rejected.append(dict(period=t,axis=pi,x_start=int(p[lo,0]),y_start=int(p[lo,1]),x_end=int(p[hi-1,0]),y_end=int(p[hi-1,1]),skeleton_pixels=hi-lo,past_coverage=coverage,learned_cross_period_affinity=affinity,left_anchor_coverage=lc,right_anchor_coverage=rc,left_anchor_affinity=la,right_anchor_affinity=ra,future_coverage=fc,future_affinity=fa,reason='insufficient same-root historical confirmation'))
    green_anchors=[(a0,b0) for a0,b0,c0 in runs((labels==2).astype(np.int32)) if c0 and b0-a0>=8]
    for (left0,left1),(right0,right1) in zip(green_anchors[:-1],green_anchors[1:]):
     lo,hi=left1,right0
     if hi<=lo or not (labels[lo:hi]==3).any():continue
     if newaxis or ((labels[lo:hi]!=3).sum()<=3 and green>=24):
      q=p[lo:hi];free=seed[q[:,1],q[:,0]]==0;q=q[free];seed[q[:,1],q[:,0]]=2;reason_seed[q[:,1],q[:,0]]=1
      if len(q):edits.append(dict(period=t,predicted_axis=pi,from_class=-1,to_class=2,skeleton_pixels=len(q),x_start=int(q[0,0]),y_start=int(q[0,1]),x_end=int(q[-1,0]),y_end=int(q[-1,1]),past_aligned_support=past,later_aligned_support=later,learned_new_probability=newprob,reason='whole extension-colored interval between unchanged new-root anchors on one reconstructed root axis'))
    for lo,hi,c in runs(labels):
     if c not in [2,3]:continue
     target=0;why=''
     left,lp=stable(labels[max(0,lo-24):lo],False);right,rp=stable(labels[hi:min(len(labels),hi+24)],True)

     if lo==0:
      values={hub_semantic[j] for j in endpoints[0] if j in hub_semantic}
      if len(values)==1:left=next(iter(values))
     if hi==len(labels):
      values={hub_semantic[j] for j in endpoints[1] if j in hub_semantic}
      if len(values)==1:right=next(iter(values))
     if c==3 and left==right and left in [1,2]:target=left;why='entire internal extension-colored segment bounded by same-root unchanged anchors'
     if c==3 and newaxis:target=2;why='new root: learned state/green shaft plus absence of aligned earlier root; extension cannot be a second birth state'
     if c==2 and left==right==1 and past>=.6 and hi-lo<=32:target=1;why='short new-colored temporal jump inside an established axis, with aligned earlier presence'
     if target:
      q=p[lo:hi];q=q[seed[q[:,1],q[:,0]]==0]
      if not len(q):continue
      seed[q[:,1],q[:,0]]=target;reason_seed[q[:,1],q[:,0]]=1;changed+=len(q);edits.append(dict(period=t,predicted_axis=pi,from_class=c,to_class=target,skeleton_pixels=len(q),x_start=int(q[0,0]),y_start=int(q[0,1]),x_end=int(q[-1,0]),y_end=int(q[-1,1]),past_aligned_support=past,later_aligned_support=later,learned_new_probability=newprob,reason=why))
    paths_out.append(dict(period=t,axis=pi,segments=len(path['parts']),skeleton_length=len(p),past_aligned_support=past,later_aligned_support=later,learned_new_probability=newprob,green_body_pixels=green,persistent_body_pixels=purple,new_axis_confirmed=newaxis,changed_skeleton_pixels=changed))

   provisional=im.copy();ss=seed>0;provisional[ss]=seed[ss]
   def arm_type(arm):
    seg=tp['segments'][arm['segment']]['points'];q=seg if arm['end']==0 else seg[::-1];v=provisional[q[:,1],q[:,0]]

    for offset in range(min(24,len(v))):
     block=v[offset:offset+8]
     if len(block)<4:continue
     for c in [1,2,3]:
      if (block==c).sum()>=max(4,int(np.ceil(.875*len(block)))):return c
    return 0
   core_rules=[]
   for semantic_pass in range(2):
    provisional[seed>0]=seed[seed>0]
    for ji,hub in enumerate(tp['hubs']):
     if hub.get('disabled'):continue
     arms=hub['arms'];colors=[arm_type(q) for q in arms];axis_types=[];force_new=False;target=0
     if ji in hub_chosen:
      opt,pr=hub_chosen[ji];used={x for pair in opt for x in pair}
      for u,v in opt:
       ca,cb=colors[u],colors[v]
       if ca==cb and ca in [1,2,3]:axis_types.append((pr,ca))
       elif {ca,cb}=={2,3}:

        axis_types.append((pr,2))
        for k in [u,v]:
         seg=tp['segments'][arms[k]['segment']]['points'];q=seg if arms[k]['end']==0 else seg[::-1];vv=provisional[q[:,1],q[:,0]]
         for lo,hi,c in runs(vv):
          if c==3 and lo<24:
           pp=q[lo:hi];seed[pp[:,1],pp[:,0]]=2;reason_seed[pp[:,1],pp[:,0]]=1
       elif ca in [1,2,3] and cb in [1,2,3]:axis_types.extend([(pr*.5,ca),(pr*.5,cb)])
      for k in set(range(len(arms)))-used:
       if colors[k] in [1,2,3]:axis_types.append((pr,colors[k]))
      if len(arms)==3 and len(used)==2:
       child=next(iter(set(range(3))-used));u,v=opt[0]
       if colors[child]==2 and colors[u]==colors[v]==1:
        q=arms[child]['body_points'];pv=history(ti,q)
        if max(pv[:ti])<.25:force_new=True;target=2
     if not axis_types:axis_types=[(.5,c) for c in colors if c in [1,2,3]]
     allowed_types={c for _,c in axis_types}
     if allowed_types:
      if not target:target=max(axis_types)[1]
      q=hub['points'];yy,xx=q[:,1],q[:,0];invalid=np.isin(im[yy,xx],[1,2,3])&~np.isin(im[yy,xx],list(allowed_types))
      if force_new:invalid|=np.isin(im[yy,xx],[1,2,3])
      core_rules.append((hub['points'],allowed_types,target,force_new));q=q[invalid];seed[q[:,1],q[:,0]]=target;reason_seed[q[:,1],q[:,0]]=2

   support_seed=seed.copy();interior=np.zeros(im.shape,np.uint8)
   for ss in tp['segments']:
    if ss.get('absorbed'):continue
    q=ss['points'];values=np.unique(seed[q[:,1],q[:,0]]);values=values[values>0]
    if len(values)!=1 or values[0] not in [1,2]:continue
    target=int(values[0]);v=im[q[:,1],q[:,0]].copy();edited=seed[q[:,1],q[:,0]]>0;v[edited]=seed[q[edited,1],q[edited,0]]
    if any(c!=target and hi-lo>=6 for lo,hi,c in runs(v)):continue
    support_seed[q[:,1],q[:,0]]=target
    if len(q)>12:inside=q[5:-5];interior[inside[:,1],inside[:,0]]=target
   sk=tp['sk'];_,near=ndi.distance_transform_edt(~sk,return_indices=True);proposal=seed[tuple(near)];reasons=reason_seed[tuple(near)];width_proposal=support_seed[tuple(near)];out=im.copy();take=(owners[t]>=0)&(proposal>0)&(np.isin(im,[2,3])|(reasons==2));out[take]=proposal[take];halo=(owners[t]>=0)&(im==3)&(width_proposal>0);out[halo]=width_proposal[halo];allowed=take|halo

   for q,types,target,force_new in core_rules:
    x0=max(0,int(q[:,0].min())-2);x1=min(2541,int(q[:,0].max())+3);y0=max(0,int(q[:,1].min())-2);y1=min(1108,int(q[:,1].max())+3);yy,xx=np.mgrid[y0:y1,x0:x1];distance=np.min((xx[:,:,None]-q[:,0])**2+(yy[:,:,None]-q[:,1])**2,axis=2);region=(owners[t][y0:y1,x0:x1]>=0)&(distance<=4);block=out[y0:y1,x0:x1];bad=region&np.isin(block,[1,2,3])&~np.isin(block,list(types));block[bad]=target;allowed[y0:y1,x0:x1]|=bad

   labels,num=ndi.label(owners[t]>=0,np.ones((3,3),bool));sizes=np.bincount(labels.ravel());trees_by={}
   for c in [1,2]:
    yy,xx=np.where(interior==c)
    if len(xx):trees_by[c]=cKDTree(np.c_[xx,yy])
   for component in np.flatnonzero((sizes>0)&(sizes<=3)):
    if not component:continue
    yy,xx=np.where(labels==component)
    if not (out[yy,xx]==3).all():continue
    xy=np.c_[xx,yy];dist={c:tr.query(xy)[0].mean() for c,tr in trees_by.items()}
    if not dist:continue
    target=min(dist,key=dist.get);other=min([v for c,v in dist.items() if c!=target] or [np.inf])
    if dist[target]<=3.5 and other-dist[target]>=2:out[yy,xx]=target;allowed[yy,xx]=True
  else:out=im.copy()
  changed=out!=im;assert np.array_equal(out>0,im>0);assert not changed[~allowed].any();assert ti or np.array_equal(out,im)
  Image.fromarray(out).save(dest/'full'/f'{t}_classes.png');Image.fromarray(changed.astype(np.uint8)*255).save(dest/'changes'/f'{t}_edited.png');Image.fromarray(allowed.astype(np.uint8)*255).save(dest/'changes'/f'{t}_allowed.png');audits.append(dict(period=t,root_pixels=int((im>0).sum()),complete_axis_paths=len(paths),junctions=sum(not hh.get('disabled') for hh in tp['hubs']),accepted_pairings=len(decisions),unresolved_pairings=uncertain,edited_pixels=int(changed.sum()),outside_support_edits=int(changed[~allowed].sum()),binary_identical=True))
  print('DECODE',t,'edits',changed.sum(),'accepted junctions',len(decisions),'of',sum(not hh.get('disabled') for hh in tp['hubs']),flush=True)
 csvout(dest/'Rejected_ambiguous_segments.csv',rejected);csvout(dest/'Complete_segment_corrections.csv',edits);csvout(dest/'Junction_pairing_audit.csv',junctions);csvout(dest/'Axis_temporal_support.csv',paths_out);csvout(dest/'Pixel_change_audit.csv',audits);js(dest/'decoding_verification.json',dict(passed=True,prototype=a.prototype,whole_field_all_four_periods=True,geometry_unchanged=True,non_anomaly_body_classes_unchanged=True,no_reference_IDs_or_classes_at_prediction=True,total_edits=sum(r['edited_pixels'] for r in audits),whole_segment_changes=len(edits),all_prior_results_preserved=True,connection_head_confidence=confidence,geometry_weight=weight,temporal_identity_threshold=identity_threshold,temporal_residual_motion_search_pixels=32))

if __name__=='__main__':main()

from paths import ROOT,WORK
"""Explicit unbranched root segments and junction-arm continuation candidates."""

import numpy as np

from scipy import ndimage as ndi

from skimage.morphology import skeletonize

def build(mask):
 sk=skeletonize(mask);yx=np.argwhere(sk);lookup={tuple(p):i for i,p in enumerate(yx)};adj=[[] for _ in yx]
 for i,(y,x) in enumerate(yx):
  for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
   j=lookup.get((y+dy,x+dx))
   if j is None:continue
   if dy and dx and ((y+dy,x) in lookup or (y,x+dx) in lookup):continue
   adj[i].append(j)
 junction=np.array([len(a)>=3 for a in adj]);parent=np.arange(len(yx))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[b]=a
 for i in np.flatnonzero(junction):
  for j in adj[i]:
   if junction[j]:union(int(i),j)

 core=junction.copy()
 for start in np.flatnonzero(junction):
  for nxt in adj[start]:
   if junction[nxt]:continue
   path=[nxt];prev,cur=int(start),nxt
   while len(path)<=2 and len(adj[cur])==2:
    follow=next(v for v in adj[cur] if v!=prev)
    if junction[follow]:
     union(int(start),follow);core[path]=True
     for v in path:parent[v]=find(int(start))
     break
    path.append(follow);prev,cur=cur,follow
 coregroups={}
 for i in np.flatnonzero(core):coregroups.setdefault(find(int(i)),[]).append(int(i))
 hubs=[];hubof=np.full(len(yx),-1,np.int32)
 for nodes in coregroups.values():
  k=len(hubs);hubof[nodes]=k;hubs.append(dict(points=yx[nodes][:,::-1],center=yx[nodes][:,::-1].mean(0),arms=[]))
 seen=set();segments=[];segof=np.full(len(yx),-1,np.int32)
 for start in [i for i in range(len(yx)) if not core[i] and sum(not core[j] for j in adj[i])<=1]+list(range(len(yx))):
  if core[start] or start in seen:continue
  path=[];prev=-1;cur=start
  while cur not in seen:
   seen.add(cur);path.append(cur);nexts=[j for j in adj[cur] if not core[j] and j!=prev and j not in seen]
   if not nexts:break
   prev,cur=cur,nexts[0]
  k=len(segments);segof[path]=k;points=yx[path][:,::-1];segments.append(dict(points=points,end_hubs=[[],[]],true_tip=[False,False]))
  for end,idx in [(0,path[0]),(1,path[-1])]:
   h=sorted({int(hubof[j]) for j in adj[idx] if core[j]});segments[-1]['end_hubs'][end]=h;segments[-1]['true_tip'][end]=len(adj[idx])==1
   for hub in h:
    q=points if end==0 else points[::-1];v=q[min(10,len(q)-1)].astype(float)-q[0];v/=max(np.linalg.norm(v),1e-8);hubs[hub]['arms'].append(dict(segment=k,end=end,point=q[0],direction=v,body_points=q[:24]))

 at=np.full(mask.shape,-1,np.int32);take=np.flatnonzero(segof>=0);at[tuple(yx[take].T)]=segof[take]
 if len(take):
  _,nearest=ndi.distance_transform_edt(at<0,return_indices=True);owner=at[tuple(nearest)];owner[~mask]=-1
 else:owner=np.full(mask.shape,-1,np.int32)
 coremap=np.full(mask.shape,-1,np.int32)
 for j,h in enumerate(hubs):p=h['points'];coremap[p[:,1],p[:,0]]=j
 return dict(sk=sk,segments=segments,hubs=hubs,owner=owner,coremap=coremap)

def matchings(n):
 if n==3:return [[(0,1)],[(0,2)],[(1,2)]]
 if n==4:return [[(0,1),(2,3)],[(0,2),(1,3)],[(0,3),(1,2)]]
 return []

def routes(topo,decisions):
 seg=topo['segments'];endlink={};accepted=[];parent=np.arange(len(seg))
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 for j,pairs,confidence in sorted(decisions,key=lambda r:-r[2]):
  arms=topo['hubs'][j]['arms']
  for a,b in pairs:
   aa,bb=arms[a],arms[b];u,v=(aa['segment'],aa['end']),(bb['segment'],bb['end']);x,y=find(u[0]),find(v[0])
   if u in endlink or v in endlink or x==y:continue
   endlink[u]=(v,j,confidence);endlink[v]=(u,j,confidence);parent[y]=x;accepted.append((j,a,b,confidence))
 paths=[];used={i for i,s in enumerate(seg) if s.get('absorbed')};rootof=np.full(len(seg),-1,np.int32)
 for start in range(len(seg)):
  if start in used:continue

  cur=start;enter=0;walked=set()
  while (cur,enter) in endlink and (cur,enter) not in walked:
   walked.add((cur,enter));(other,end),hub,c=endlink[(cur,enter)];cur=other;enter=1-end
  path=[];links=[]
  while cur not in used:
   used.add(cur);rootof[cur]=len(paths);path.append((cur,enter));exit=1-enter
   if (cur,exit) not in endlink:break
   (nxt,end),hub,c=endlink[(cur,exit)];links.append((hub,c));cur=nxt;enter=end
  paths.append(dict(parts=path,links=links))
 return paths,rootof,accepted

def fuse_crossing_bridges(topo):
 """A thick X can skeletonize as two Y nodes joined by a tiny shared stripe."""
 hubs=topo['hubs'];segments=topo['segments'];records=[]
 for si,s in sorted(enumerate(segments),key=lambda v:len(v[1]['points'])):
  if len(s['points'])>8 or len(s['points'])<1 or len(s['end_hubs'][0])!=1 or len(s['end_hubs'][1])!=1:continue
  a,b=s['end_hubs'][0][0],s['end_hubs'][1][0]
  if a==b or hubs[a].get('disabled') or hubs[b].get('disabled') or len(hubs[a]['arms'])!=3 or len(hubs[b]['arms'])!=3:continue
  external=[x.copy() for j in [a,b] for x in hubs[j]['arms'] if x['segment']!=si]
  if len(external)!=4 or any(len(x['body_points'])<4 for x in external):continue
  possibilities=matchings(4);alignment=[min(-external[i]['direction']@external[j]['direction'] for i,j in opt) for opt in possibilities]
  if max(alignment)<.82:continue

  new=len(hubs);points=np.concatenate([hubs[a]['points'],s['points'],hubs[b]['points']]);hubs.append(dict(points=points,center=points.mean(0),arms=external,fused_short_bridge=si,source_hubs=[a,b]));hubs[a]['disabled']=True;hubs[b]['disabled']=True;s['absorbed']=True
  for arm in external:
   end=arm['end'];ss=segments[arm['segment']];ss['end_hubs'][end]=[new if j in [a,b] else j for j in ss['end_hubs'][end]]
  records.append(dict(new_hub=new,source_hubs=[a,b],absorbed_segment=si,shared_skeleton_pixels=len(s['points']),best_minimum_opposing_alignment=float(max(alignment))))
 return records

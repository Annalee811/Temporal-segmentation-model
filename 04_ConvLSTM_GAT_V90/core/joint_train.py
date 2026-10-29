from paths import ROOT,WORK
"""Joint history/graph training for root continuation and stable chronological root state."""

import json,pickle,time,os

import numpy as np

import torch

from torch.nn import functional as F

from model import LocalCorrector

from data import Data

from backbone import pool_groups

from train import PairHead,matching_metrics

from paths import OUT,LAST,SEED,js,sha,csvout

def save(p,v):q=p.with_suffix('.tmp');torch.save(v,q);os.replace(q,p)

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(SEED);data=Data();d=dict(np.load(OUT/'data/pairs.npz'))
 with (OUT/'data/topology.pkl').open('rb') as f:top=pickle.load(f)
 model=LocalCorrector();base=torch.load(LAST/'best.pt',weights_only=False,map_location='cpu');assert sha(LAST/'best.pt')==json.loads((LAST/'initialization.json').read_text())['checkpoint_sha256'];model.load_state_dict(base['state_dict'])
 for module in [model.pixel,model.local,model.edit,model.branch]:
  for p in module.parameters():p.requires_grad_(False)
 pair=PairHead(d['x'].shape[1]);pair.load_state_dict(torch.load(OUT/'best_pair_head.pt',weights_only=True)['state_dict']);opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad],lr=.0001),dict(params=pair.parameters(),lr=.0005)],weight_decay=.0005);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100,eta_min=3e-6)
 armnode=[];armptr=[0];pair_a=np.zeros(len(d['x']),np.int64);pair_b=pair_a.copy()
 groups_by={(q['period'],q['hub']):q for q in top['groups']}
 for t,items in top['arms'].items():
  for hub,arms in items:
   ids=[]
   for a in arms:ids.append(len(armptr)-1);armnode.extend(a['nodes']);armptr.append(len(armnode))
   g=groups_by[(t,hub)]
   for key,k in g['pair_indices'].items():a,b=map(int,key.split(','));pair_a[k]=ids[a];pair_b[k]=ids[b]
 an=torch.tensor(armnode);ptr=torch.tensor(armptr);A=torch.tensor(pair_a);B=torch.tensor(pair_b);mu=torch.from_numpy(d['mean']);sd=torch.from_numpy(d['std']);fixed=torch.from_numpy(d['raw'][:,96:]);y=torch.tensor(d['target'],dtype=torch.float32);tr=np.flatnonzero(d['split']==0);va=np.flatnonzero(d['split']==1);pos=torch.tensor(float((y[tr]==0).sum()/(y[tr]==1).sum()))
 ranking=[g for g in top['groups'] if g['split']==0 and g['target_option']>=0];idx=np.zeros((len(ranking),3,2),np.int64);mask=np.zeros(idx.shape,np.float32);labels=[]
 for k,g in enumerate(ranking):
  labels.append(g['target_option'])
  for j,opt0 in enumerate(g['options']):
   for m,(a,b) in enumerate(opt0):idx[k,j,m]=g['pair_indices'][str(a)+','+str(b)];mask[k,j,m]=1
 idx=torch.tensor(idx);mask=torch.tensor(mask);labels=torch.tensor(labels)
 rootaudit=json.loads((data.folder/'root_state_audit.json').read_text())['groups'];roots={(g['root_target'],g['period']):i for i,g in enumerate(rootaudit)};temporal=[]
 for (rid,t),k in roots.items():
  for prev in ['T2','T3']:
   if (prev,t) in [('T2','T3'),('T3','T4')] and (rid,prev) in roots and rootaudit[k]['state_new']==0:temporal.append([roots[(rid,prev)],k])
 temporal=torch.tensor(temporal,dtype=torch.int64).reshape(-1,2);rs=data.rs;linktr={same:np.flatnonzero((data.g['split']==0)&(data.same==same)) for same in [True,False]};linktarget=torch.tensor(data.g['target'],dtype=torch.float32);history=[];best=-1.;tic=time.monotonic()
 def features(z):
  e=F.normalize(pool_groups(z,an,ptr),dim=1);x=torch.cat([(e[A]-e[B]).abs(),e[A]*e[B],fixed],1);return (x-mu)/sd
 def snapshot(epoch):return dict(epoch=epoch,backbone=model.state_dict(),pair=pair.state_dict(),source_checkpoint_sha256=sha(LAST/'best.pt'))
 first=1
 if (OUT/'joint_resume.pt').exists():
  c=torch.load(OUT/'joint_resume.pt',weights_only=False,map_location='cpu');model.load_state_dict(c['backbone']);pair.load_state_dict(c['pair']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);torch.set_rng_state(c['rng']);history=c['history'];best=c['best'];first=c['epoch']+1
 else:save(OUT/'joint_initial.pt',snapshot(0))
 for epoch in range(first,101):
  if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('User stop marker; whole completed epochs saved')
  st=time.monotonic();model.train();pair.train();opt.zero_grad(set_to_none=True);h,links,z=data.forward(model);score=pair(features(z));bce=F.binary_cross_entropy_with_logits(score[tr],y[tr],pos_weight=pos);rank=F.cross_entropy((score[idx]*mask).sum(-1),labels);identity=[]
  for same in [True,False]:
   ix=linktr[same];target=linktarget[ix];pw=torch.tensor(float((target==0).sum()/(target==1).sum()));identity.append(F.binary_cross_entropy_with_logits(links[ix],target,pos_weight=pw))
  state=model.root_state(pool_groups(h,rs['nodes'],rs['ptr']))[:,0];state_loss=F.binary_cross_entropy_with_logits(state,rs['target'].float());newprob=state.sigmoid();consistency=F.relu(newprob[temporal[:,1]]-newprob[temporal[:,0]]).square().mean() if len(temporal) else state.sum()*0;loss=bce+.5*rank+.2*torch.stack(identity).mean()+.2*state_loss+.1*consistency;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(pair.parameters()),5,error_if_nonfinite=True)
  if epoch==1:
   mem=float(model.memory.gates.weight.grad.abs().sum());gat=[float(q.project.weight.grad.abs().sum()) for q in model.layers];assert mem>0 and all(v>0 for v in gat);js(OUT/'joint_gradient_verification.json',dict(passed=True,ConvLSTM_gradient=mem,GAT_gradients=gat,source24_confirmed_root_supervision=True,learned_identity_gates_graph_messages=True))
  opt.step();sched.step();model.eval();pair.eval()
  with torch.inference_mode():hh,ll,zz=data.forward(model);s=pair(features(zz));vl=float(F.binary_cross_entropy_with_logits(s[va],y[va]));acc=matching_metrics(top['groups'],s.numpy(),d['raw'],1,2.)
  value=acc['accuracy']-.001*vl
  if value>best:best=value;save(OUT/'joint_best.pt',snapshot(epoch))
  history.append(dict(epoch=epoch,training_total=float(loss.detach()),training_continuation_BCE=float(bce.detach()),training_junction_ranking=float(rank.detach()),training_temporal_state_consistency=float(consistency.detach()),validation_pair_BCE=vl,validation_junction_assignment_accuracy=acc['accuracy'],validation_junctions=acc['available'],learning_rate=opt.param_groups[0]['lr'],seconds=time.monotonic()-st));csvout(OUT/'joint_training_history.csv',history);save(OUT/'joint_resume.pt',dict(**snapshot(epoch),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=torch.get_rng_state(),history=history,best=best));js(OUT/'joint_status.json',dict(stage='joint_ConvLSTM_GAT_continuation_training',epoch_completed=epoch,epochs_total=100,pid=os.getpid(),last_epoch_seconds=history[-1]['seconds']))
  if epoch==1 or epoch%10==0:print('JOINT_EPOCH',epoch,'val junction matching',acc,'seconds',history[-1]['seconds'],flush=True)
 c=torch.load(OUT/'joint_best.pt',weights_only=False,map_location='cpu');model.load_state_dict(c['backbone']);pair.load_state_dict(c['pair']);model.eval();pair.eval()
 with torch.inference_mode():h,link,z=data.forward(model);x=features(z);logit=pair(x).numpy()
 weight=max([0.,1.,2.,4.,8.],key=lambda w:(matching_metrics(top['groups'],logit,d['raw'],1,w)['accuracy'],-w));choices=[]
 for conf in [.5,.6,.7,.8,.9,.95]:
  m=matching_metrics(top['groups'],logit,d['raw'],1,weight,conf)
  if m['accepted']>=10:choices.append((m['accuracy']>=.9,m['coverage'] if m['accuracy']>=.9 else m['accuracy'],conf))
 confidence=max(choices)[2] if choices else .9;np.save(OUT/'joint_pair_logits.npy',logit);np.savez_compressed(OUT/'joint_inference.npz',h=h.numpy(),embedding=z.numpy(),connection=link.sigmoid().numpy(),features=x.numpy())
 js(OUT/'joint_training_results.json',dict(epochs=100,selected_epoch=c['epoch'],geometry_weight=weight,assignment_confidence=confidence,validation=matching_metrics(top['groups'],logit,d['raw'],1,weight,confidence),test=matching_metrics(top['groups'],logit,d['raw'],2,weight,confidence),test_all=matching_metrics(top['groups'],logit,d['raw'],2,weight),trained_ConvLSTM=True,trained_GAT=True,learned_relation_gates_attention=True,pair_head_ranking=True,root_state_temporal_consistency=True,temporal_train_pairs=len(temporal),checkpoint_sha256=sha(OUT/'joint_best.pt'),elapsed_seconds=time.monotonic()-tic));print('JOINT_COMPLETE',flush=True)

if __name__=='__main__':main()

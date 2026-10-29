from paths import ROOT,WORK
"""Learn root-arm continuation directly, rather than nearby-node/global pixel classification."""

import json,pickle,time

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

from paths import OUT,SEED,js,sha,csvout

class PairHead(nn.Module):
 def __init__(self,n):super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.ELU(),nn.Linear(64,32),nn.ELU(),nn.Linear(32,1))
 def forward(self,x):return self.net(x)[:,0]

def group_score(group,logit,raw,geometry_weight):
 score=[]
 for opt in group['options']:
  ids=[group['pair_indices'][str(i)+','+str(j)] for i,j in opt];score.append(float(logit[ids].sum()+geometry_weight*raw[ids,96].sum()))
 return np.array(score)

def matching_metrics(groups,logit,raw,split,geometry_weight,confidence=0):
 total=correct=0;alln=0
 for g in groups:
  if g['split']!=split or g['target_option']<0:continue
  alln+=1;s=group_score(g,logit,raw,geometry_weight);p=np.exp(s-s.max());p/=p.sum();i=int(s.argmax())
  if p[i]<confidence:continue
  total+=1;correct+=i==g['target_option']
 return dict(correct=correct,accepted=total,available=alln,accuracy=correct/max(total,1),coverage=total/max(alln,1))

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(SEED);d=dict(np.load(OUT/'data/pairs.npz'));x=torch.from_numpy(d['x']);y=torch.tensor(d['target'],dtype=torch.float32);tr=np.flatnonzero(d['split']==0);va=np.flatnonzero(d['split']==1);te=np.flatnonzero(d['split']==2)
 with (OUT/'data/topology.pkl').open('rb') as f:groups=pickle.load(f)['groups']
 model=PairHead(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,eta_min=1e-5);hist=[];best=-1.;tic=time.monotonic();pos=torch.tensor(float((y[tr]==0).sum()/(y[tr]==1).sum()))

 ranking=[g for g in groups if g['split']==0 and g['target_option']>=0]
 for ep in range(101):
  if ep:
   model.train();optimizer.zero_grad();z=model(x);bce=F.binary_cross_entropy_with_logits(z[tr],y[tr],pos_weight=pos);ranks=[]
   for g in ranking:
    logits=torch.stack([z[[g['pair_indices'][str(i)+','+str(j)] for i,j in opt]].sum() for opt in g['options']]);ranks.append(F.cross_entropy(logits[None],torch.tensor([g['target_option']])))
   rank=torch.stack(ranks).mean();loss=bce+.5*rank;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();schedule.step()
  model.eval()
  with torch.no_grad():z=model(x);vl=float(F.binary_cross_entropy_with_logits(z[va],y[va]));train=float(F.binary_cross_entropy_with_logits(z[tr],y[tr]));a=matching_metrics(groups,z.numpy(),d['raw'],1,2.)
  hist.append(dict(epoch=ep,training_pair_BCE=train,validation_pair_BCE=vl,validation_matching_accuracy=a['accuracy'],validation_matching_junctions=a['available'],learning_rate=optimizer.param_groups[0]['lr']))
  score=a['accuracy']-.001*vl
  if ep and score>best:best=score;torch.save(dict(state_dict=model.state_dict(),epoch=ep,feature_dimension=x.shape[1]),OUT/'best_pair_head.pt')
  if ep%20==0:print('PAIR_EPOCH',ep,'val matching',a,flush=True)
  csvout(OUT/'pair_training_history.csv',hist)
 ck=torch.load(OUT/'best_pair_head.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state_dict']);model.eval()
 with torch.no_grad():logit=model(x).numpy()
 weight=max([0.,1.,2.,4.,8.],key=lambda w:(matching_metrics(groups,logit,d['raw'],1,w)['accuracy'],-w))

 choices=[]
 for c in [.5,.6,.7,.8,.9,.95]:
  m=matching_metrics(groups,logit,d['raw'],1,weight,c)
  if m['accepted']>=10:choices.append((m['accuracy']>=.9,m['coverage'] if m['accuracy']>=.9 else m['accuracy'],c,m))
 if choices:chosen=max(choices,key=lambda r:(r[0],r[1],r[2]));confidence=chosen[2]
 else:confidence=.9
 np.save(OUT/'pair_logits.npy',logit);js(OUT/'pair_training_results.json',dict(epochs=100,selected_epoch=ck['epoch'],geometry_weight=weight,assignment_confidence=confidence,validation=matching_metrics(groups,logit,d['raw'],1,weight,confidence),test=matching_metrics(groups,logit,d['raw'],2,weight,confidence),test_all=matching_metrics(groups,logit,d['raw'],2,weight),elapsed_seconds=time.monotonic()-tic,checkpoint_sha256=sha(OUT/'best_pair_head.pt'),trained_scope='New continuation head on frozen learned ConvLSTM/GAT embeddings; no U-Net or full temporal CNN retraining'))
 print('PAIR_COMPLETE',json.loads((OUT/'pair_training_results.json').read_text()),flush=True)

if __name__=='__main__':main()

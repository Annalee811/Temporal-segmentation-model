"""Warm-start V35 U-Net with binary cross-tube labels; grouped evaluation then all-label refit."""
import argparse,csv,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import RootUNet,objective
import common as c

HERE=Path(__file__).resolve().parent

def js(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');torch.save(d,q);q.replace(p)
def metrics(total):
 d=c.metrics(total);tp,fp,fn,tn=map(int,total);d.update(Accuracy=(tp+tn)/(tp+fp+fn+tn),Specificity=tn/(tn+fp) if tn+fp else None,Balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn)*(tn+fp) else None);return d

def main(a):
 torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 out=a.output.resolve();assets=HERE/'assets';original=out/'weights/V35_original_epoch77.pt';initial_hash=c.sha(original)
 old=[];new=[]
 for r in c.csvread(assets/'data/manifest.csv'):
  im=np.array(Image.open(assets/r['RGB_path'][5:]).convert('RGB')) if False else np.array(Image.open(assets/r['RGB_path']).convert('RGB'))
  gt=np.array(Image.open(assets/r['mask_path']).convert('L'));old.append(dict(id=r['image_id'],split=r['split'],source='original',tube='0353-2',im=im,gt=gt))
 for r in c.csvread(out/'data/manifest.csv'):
  im=np.array(Image.open(out/'data/RGB'/r['image_id']).convert('RGB'));gt=np.array(Image.open(out/'data/mask'/r['image_id']).convert('L'));assert im.shape[:2]==gt.shape==(500,500)
  new.append(dict(id=r['image_id'],split=r['split'],source='cross_tube',tube=r['tube'],im=im,gt=gt))
 groups={s:{r['tube'] for r in new if r['split']==s} for s in ['train','val','test']}
 assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
 config=dict(seed=a.seed,maximum_development_epochs=a.epochs,minimum_development_epochs=6,early_stop_patience=5,initial_lr=5e-5,weight_decay=1e-4,batch_size=2,gradient_accumulation=4,threshold=.5,patch_size=256,inference_stride=192,model='V35 ResNet18 U-Net (unchanged architecture)',initialization_SHA256=initial_hash,selection='mean of cross-tube validation Dice and original validation Dice; test untouched until selection',new_split_tubes={s:sorted(v) for s,v in groups.items()},production='Restart original V35 weights and train original 200 train images plus all 66 new images for development-selected epoch count; original val/test regions excluded from new training masks',reference_limit='Prediction-initialized user edits; agreement with supplied reference is not a de novo fully independent annotation study.')
 js(out/'training_config.json',config)
 model=RootUNet();model.load_state_dict(torch.load(original,map_location='cpu',weights_only=True)['state_dict'])
 def tensors(items,aug=False):
  xx=[];yy=[];vv=[]
  for r in items:
   im=r['im'];gt=r['gt'];h,w=gt.shape
   if aug and h>256:
    x=random.randrange(w-255);y=random.randrange(h-255);im=im[y:y+256,x:x+256];gt=gt[y:y+256,x:x+256]
   if aug:
    k=random.randrange(4);im=np.rot90(im,k);gt=np.rot90(gt,k)
    if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt)
    if random.random()<.5:im=np.flipud(im);gt=np.flipud(gt)
   im=im.astype(np.float32)/255
   if aug:im=np.clip(im*random.uniform(.9,1.1),0,1)**random.uniform(.9,1.1)
   xx.append(im.copy());yy.append((gt==255).astype('float32'));vv.append((gt!=128).astype('float32'))
  return torch.from_numpy(np.stack(xx).transpose(0,3,1,2).copy()),torch.from_numpy(np.stack(yy)[:,None]),torch.from_numpy(np.stack(vv)[:,None])
 @torch.inference_mode()
 def predict(im):
  model.eval();h,w=im.shape[:2];xs=list(range(0,w-255,192));ys=list(range(0,h-255,192));xs=sorted(set(xs+[w-256]));ys=sorted(set(ys+[h-256]));xy=[(x,y) for y in ys for x in xs]
  win=np.maximum(np.hanning(256)[:,None]*np.hanning(256)[None,:],.02).astype('float32');sm=np.zeros((h,w),np.float32);wg=np.zeros_like(sm)
  for i in range(0,len(xy),2):
   cc=xy[i:i+2];x=torch.from_numpy(np.stack([im[y:y+256,x:x+256] for x,y in cc]).transpose(0,3,1,2).copy()).float()/255
   pp=model(x).sigmoid()[:,0].numpy()
   for (x,y),p in zip(cc,pp):sm[y:y+256,x:x+256]+=p*win;wg[y:y+256,x:x+256]+=win
  return sm/wg
 def evaluate(rows,folder=None):
  counts=np.zeros(4,np.int64);per=[];losses=[]
  for r in rows:
   p=predict(r['im']);cc=c.counts(r['gt'],p>=.5);counts+=cc;per.append(dict(image_id=r['id'],tube=r['tube'],**metrics(cc)))
   valid=r['gt']!=128;g=(r['gt']==255).astype('float32');pv=np.clip(p[valid],1e-6,1-1e-6);gv=g[valid];losses.append(float(-(gv*np.log(pv)+(1-gv)*np.log(1-pv)).mean()+1-(2*(pv*gv).sum()+1)/(pv.sum()+gv.sum()+1)))
   if folder:
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);np.save(folder/(r['id']+'_probability.npy'),p);Image.fromarray((p>=.5).astype('uint8')*255).save(folder/(r['id']+'_root.png'))
  return metrics(counts),float(np.mean(losses)),per
 oldtrain=[r for r in old if r['split']=='train'];oldval=[r for r in old if r['split']=='val'];oldtest=[r for r in old if r['split']=='test'];newval=[r for r in new if r['split']=='val'];newtest=[r for r in new if r['split']=='test']
 baseline=[]
 for name,rr in [('original_validation',oldval),('cross_tube_validation',newval),('original_internal_test',oldtest),('cross_tube_test',newtest)]:
  m,l,per=evaluate(rr,out/'baseline_predictions'/name);baseline.append(dict(model='V35_original',split=name,images=len(rr),loss=l,**m));csvout(out/'metrics'/('baseline_'+name+'_per_image.csv'),per)
 csvout(out/'baseline_metrics.csv',baseline);print('BASELINE',baseline,flush=True)
 def run(stage,training,epochs,early):
  torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);model.load_state_dict(torch.load(original,map_location='cpu',weights_only=True)['state_dict']);opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3,min_lr=1e-6)
  hist=[];best=-1;bestepoch=0;bad=0;start=1;folder=out/stage;folder.mkdir(exist_ok=True)
  resume=folder/'last_training_state.pt'
  if resume.exists():
   d=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(d['state_dict']);opt.load_state_dict(d['optimizer']);sched.load_state_dict(d['scheduler']);hist=d['history'];best=d['best'];bestepoch=d['best_epoch'];bad=d['bad'];start=d['epoch']+1;torch.set_rng_state(d['torch_rng']);random.setstate(d['random_rng']);np.random.set_state(d['numpy_rng']);print('RESUME',stage,start,flush=True)
  else:
   m1,l1,_=evaluate(oldval);m2,l2,_=evaluate(newval);hist=[dict(epoch=0,train_optimization_loss=None,original_validation_Dice=m1['Dice'],cross_tube_validation_Dice=m2['Dice'],validation_score=(m1['Dice']+m2['Dice'])/2,original_validation_loss=l1,cross_tube_validation_loss=l2,lr=5e-5,seconds=0.)];best=hist[0]['validation_score'];save(folder/'best_model.pt',dict(state_dict=model.state_dict()));csvout(folder/'history.csv',hist)
  for epoch in range(start,epochs+1):
   tic=time.time();model.train();model.encoder.eval() # retain pretrained BN running statistics during small-batch adaptation
   work=oldtrain+training*2;random.shuffle(work);opt.zero_grad(set_to_none=True);ls=0.;lr=opt.param_groups[0]['lr'];steps=(len(work)+1)//2
   for step in range(steps):
    rr=work[step*2:step*2+2];x,y,v=tensors(rr,True)
    with torch.autocast('cpu',dtype=torch.bfloat16):z=model(x)
    loss=objective(z,y,v);assert torch.isfinite(loss);group=min(4,steps-(step//4)*4);(loss/group).backward();ls+=float(loss.detach())*len(rr)
    if (step+1)%4==0 or step+1==steps:torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();opt.zero_grad(set_to_none=True)
    if step%20==0:js(out/'status.json',dict(stage=stage,epoch=epoch,max_epochs=epochs,processed=min((step+1)*2,len(work)),training_images=len(work)))
   m1,l1,_=evaluate(oldval);m2,l2,_=evaluate(newval);score=(m1['Dice']+m2['Dice'])/2
   row=dict(epoch=epoch,train_optimization_loss=ls/len(work),original_validation_Dice=m1['Dice'],cross_tube_validation_Dice=m2['Dice'],validation_score=score,original_validation_loss=l1,cross_tube_validation_loss=l2,lr=lr,seconds=time.time()-tic);hist.append(row)
   if score>best+1e-5:best=score;bestepoch=epoch;bad=0;save(folder/'best_model.pt',dict(state_dict=model.state_dict()))
   else:bad+=1
   if early:sched.step(score)
   save(folder/'last_model.pt',dict(state_dict=model.state_dict()));save(resume,dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),history=hist,best=best,best_epoch=bestepoch,bad=bad,epoch=epoch,torch_rng=torch.get_rng_state(),random_rng=random.getstate(),numpy_rng=np.random.get_state()));csvout(folder/'history.csv',hist);print('EPOCH',stage,row,'best',bestepoch,flush=True)
   if early and epoch>=6 and bad>=5:break
  js(folder/'completion.json',dict(epochs=hist[-1]['epoch'],best_epoch=bestepoch,best_score=best,selected_checkpoint='best_model.pt' if early else 'last_model.pt',initial_SHA256=initial_hash));return bestepoch,hist
 completed=out/'development/completion.json'
 if completed.exists():bestepoch=json.loads(completed.read_text())['best_epoch']
 else:bestepoch,_=run('development',[r for r in new if r['split']=='train'],a.epochs,True)
 # Cross-tube and internal tests are scored after model selection; never tune to these outputs.
 model.load_state_dict(torch.load(out/'development/best_model.pt',weights_only=True)['state_dict']);results=baseline.copy()
 for name,rr in [('original_validation',oldval),('cross_tube_validation',newval),('original_internal_test',oldtest),('cross_tube_test',newtest)]:
  m,l,per=evaluate(rr,out/'development_predictions'/name);results.append(dict(model='V35_cross_tube_development',split=name,images=len(rr),loss=l,**m));csvout(out/'metrics'/('development_'+name+'_per_image.csv'),per)
 csvout(out/'evaluation_metrics.csv',results);print('DEVELOPMENT_TEST',results[-2:],flush=True)
 if bestepoch==0:
  save(out/'weights/V35_cross_tube_all_labels.pt',dict(state_dict=model.state_dict()));raise RuntimeError('No validation improvement; original retained; do not deploy an unvalidated refit')
 if not (out/'production/completion.json').exists():run('production',new,bestepoch,False)
 shutil_source=out/'production/last_model.pt';save(out/'weights/V35_cross_tube_all_labels.pt',torch.load(shutil_source,map_location='cpu',weights_only=True))
 model.load_state_dict(torch.load(out/'weights/V35_cross_tube_all_labels.pt',weights_only=True)['state_dict'])
 # This original internal set remains excluded from the final refit. New cross-tube images are now training data.
 m,l,per=evaluate(oldtest,out/'production_predictions/original_internal_test');csvout(out/'production_internal_retention_metrics.csv',[dict(model='V35_cross_tube_all_labels',split='original_internal_test',images=len(oldtest),loss=l,**m)]);csvout(out/'metrics/production_original_test_per_image.csv',per)
 assert c.sha(original)==initial_hash and c.sha(HERE/'assets/weights/unet/best_model.pt')==initial_hash
 js(out/'completion.json',dict(state='complete',development_selected_epoch=bestepoch,production_epochs=bestepoch,original_weights_SHA256=initial_hash,production_weights_SHA256=c.sha(out/'weights/V35_cross_tube_all_labels.pt'),all_new_images_used_in_production=len(new),development_test_images=len(newtest),new_test_independence='Held-out tubes in development only; all-label production has no independent cross-tube test remaining',original_test_excluded_from_all_training=True));js(out/'status.json',dict(state='complete'));print('COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=20261124);main(p.parse_args())

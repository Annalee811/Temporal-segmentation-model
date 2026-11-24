"""Fixed 100-epoch V35 training on a newly grouped 244+66 binary dataset."""
import argparse,csv,json,math,os,random,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from models import RootUNet,objective
import common as c

HERE=Path(__file__).resolve().parent

def js(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');torch.save(value,q);q.replace(p)
def csvout(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 q.replace(p)
def metrics(cc):
 d=c.metrics(cc);tp,fp,fn,tn=map(int,cc);d['Accuracy']=(tp+tn)/max(1,tp+fp+fn+tn);d['Specificity']=tn/(tn+fp) if tn+fp else None;d['Balanced_accuracy']=(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn)*(tn+fp) else None;return d

def main(a):
 assert a.epochs==100,'This release uses a fixed 100-epoch budget'
 out=a.output.resolve();torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 manifest=out/'data/manifest.csv';audit=c.read(out/'data_audit.json');assert c.sha(manifest)==audit['manifest_SHA256'];rows=c.csvread(manifest);assert len(rows)==310
 cache={}
 for r in rows:
  for kind,key in [('RGB','RGB_sha256'),('mask','mask_sha256')]:assert c.sha(out/'data'/kind/r['image_id'])==r[key]
  im=np.array(Image.open(out/'data/RGB'/r['image_id']).convert('RGB'));gt=np.array(Image.open(out/'data/mask'/r['image_id']).convert('L'));assert im.shape[:2]==gt.shape;cache[r['image_id']]=(im,gt)
 training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='val'];assert len(training)==248 and len(validation)==62;assert not({r['group_id'] for r in training}&{r['group_id'] for r in validation});assert {r['split'] for r in rows}=={'train','val'}
 original=out/'weights/V35_original_epoch77.pt';original_hash=c.sha(original);assert original_hash==audit['original_weights_SHA256'];model=RootUNet().to(memory_format=torch.channels_last);model.load_state_dict(torch.load(original,map_location='cpu',weights_only=True)['state_dict'])
 protocol=dict(model='V35 ResNet18 U-Net',classes='root/background only; temporal categories ignored',total_images=310,original_images=244,new_images=66,training_images=248,validation_images=62,test_images=0,independent_validation=False,initialization='warm start from original V35 selected epoch77; explicitly chosen by user',initial_weights_SHA256=original_hash,seed=a.seed,epochs=100,early_stopping=False,physical_batch=2,gradient_accumulation=4,effective_batch=8,initial_lr=5e-5,weight_decay=1e-4,lr_factor=.5,lr_patience=8,minimum_lr=1e-6,loss='original V35 hard-negative BCE + Dice + 0.15 clDice',patch_size=256,inference_stride=192,threshold=.5,encoder_running_statistics='original V35 BatchNorm running statistics held fixed; all affine and convolution weights trainable',training_coverage='Every one of the 248 training images visited once per epoch; one random native 256px crop per 500px image; no resizing',checkpoint_selection='Highest validation Dice over epochs1-100; epoch100 also saved',validation_caveat='Internal training validation, not independent validation. Original pretrained weights historically saw some old images now assigned to validation. New labels were prediction-initialized and edited by user.',manifest_SHA256=c.sha(manifest),source_code_SHA256=c.sha(Path(__file__)),runtime=dict(torch=torch.__version__,numpy=np.__version__,threads=a.threads,memory_format='channels_last'))
 protocol_path=out/'protocol.json'
 if protocol_path.exists():assert c.read(protocol_path)==protocol,'Protocol mismatch on resume'
 else:js(protocol_path,protocol)
 optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=8,min_lr=1e-6)
 def tensors(rr,augment):
  xx=[];yy=[];vv=[]
  for r in rr:
   im,gt=cache[r['image_id']];h,w=gt.shape
   if augment and (h>256 or w>256):
    x=random.randrange(w-255);y=random.randrange(h-255);im=im[y:y+256,x:x+256];gt=gt[y:y+256,x:x+256]
   if augment:
    k=random.randrange(4);im=np.rot90(im,k);gt=np.rot90(gt,k)
    if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt)
    if random.random()<.5:im=np.flipud(im);gt=np.flipud(gt)
   im=im.astype('float32')/255
   if augment:im=np.clip(im*random.uniform(.9,1.1),0,1)**random.uniform(.9,1.1)
   xx.append(im.copy());yy.append((gt==255).astype('float32'));vv.append((gt!=128).astype('float32'))
  return torch.from_numpy(np.stack(xx).transpose(0,3,1,2).copy()).to(memory_format=torch.channels_last),torch.from_numpy(np.stack(yy)[:,None]),torch.from_numpy(np.stack(vv)[:,None])
 @torch.inference_mode()
 def evaluate(rr,folder=None):
  model.eval();total=np.zeros(4,np.int64);sub=defaultdict(lambda:np.zeros(4,np.int64));per=[];loss_sum=0.;tile_count=0
  win=np.maximum(np.hanning(256)[:,None]*np.hanning(256)[None,:],.02).astype('float32')
  for r in rr:
   im,gt=cache[r['image_id']];h,w=gt.shape;xs=sorted(set(list(range(0,w-255,192))+[w-256]));ys=sorted(set(list(range(0,h-255,192))+[h-256]));xy=[(x,y) for y in ys for x in xs];sm=np.zeros((h,w),np.float32);wg=np.zeros_like(sm)
   for i in range(0,len(xy),2):
    cc=xy[i:i+2];x=torch.from_numpy(np.stack([im[y:y+256,x:x+256] for x,y in cc]).transpose(0,3,1,2).copy()).float().to(memory_format=torch.channels_last)/255;y=torch.from_numpy(np.stack([(gt[yy:yy+256,xx:xx+256]==255).astype('float32') for xx,yy in cc])[:,None]);v=torch.from_numpy(np.stack([(gt[yy:yy+256,xx:xx+256]!=128).astype('float32') for xx,yy in cc])[:,None]);z=model(x);loss=objective(z,y,v);assert torch.isfinite(loss);loss_sum+=float(loss)*len(cc);tile_count+=len(cc)
    pp=z.sigmoid()[:,0].numpy()
    for (xx,yy),p in zip(cc,pp):sm[yy:yy+256,xx:xx+256]+=p*win;wg[yy:yy+256,xx:xx+256]+=win
   p=sm/wg;counts=c.counts(gt,p>=.5);total+=counts;sub[r['source']]+=counts;per.append(dict(image_id=r['image_id'],source=r['source'],position=r['position'],period=r['period'],**metrics(counts)))
   if folder:
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);np.save(folder/(r['image_id']+'_probability.npy'),p);Image.fromarray((p>=.5).astype('uint8')*255).save(folder/(r['image_id']+'_root.png'))
  return metrics(total),loss_sum/tile_count,{s:metrics(cc) for s,cc in sub.items()},per
 resume=out/'last_training_state.pt';history=[];best=-1.;best_epoch=0;start=1
 if resume.exists():
  ck=torch.load(resume,map_location='cpu',weights_only=False);assert ck['protocol_SHA256']==c.sha(protocol_path);model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];start=ck['epoch']+1;torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);print('RESUME',start,flush=True)
 else:
  m,loss,sub,per=evaluate(validation,out/'baseline_validation_predictions');csvout(out/'baseline_validation_per_image.csv',per);csvout(out/'baseline_validation_metrics.csv',[dict(model='original_V35_reference',split='repartitioned_internal_validation',images=62,loss=loss,**m)])
  history=[dict(epoch=0,train_loss=None,validation_loss=loss,train_Dice=None,validation_Dice=m['Dice'],validation_IoU=m['IoU'],validation_Precision=m['Precision'],validation_Recall=m['Recall'],validation_Accuracy=m['Accuracy'],validation_Balanced_accuracy=m['Balanced_accuracy'],old_validation_Dice=sub['original_244']['Dice'],new_validation_Dice=sub['submitted_66']['Dice'],learning_rate=5e-5,training_views=0,optimizer_updates=0,seconds=0.)];csvout(out/'history.csv',history);print('INITIAL_VALIDATION',m,flush=True)
 for epoch in range(start,101):
  tic=time.monotonic();model.train();model.encoder.eval();order=training.copy();random.shuffle(order);optimizer.zero_grad(set_to_none=True);loss_sum=0.;counts=np.zeros(4,np.int64);steps=len(order)//2;updates=0;lr=optimizer.param_groups[0]['lr']
  for step in range(steps):
   rr=order[step*2:step*2+2];x,y,v=tensors(rr,True)
   with torch.autocast('cpu',dtype=torch.bfloat16):z=model(x)
   loss=objective(z,y,v);assert torch.isfinite(loss),'Nonfinite loss';group=min(4,steps-(step//4)*4);(loss/group).backward();loss_sum+=float(loss.detach())*2
   pp=z.detach().float().sigmoid()[:,0].numpy();gg=np.where(v[:,0].numpy()>.5,y[:,0].numpy()*255,128).astype('uint8')
   for g,p in zip(gg,pp):counts+=c.counts(g,p>=.5)
   if (step+1)%4==0 or step+1==steps:
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
   if step%20==0:js(out/'status.json',dict(state='training',epoch=epoch,completed_epochs=epoch-1,maximum_epochs=100,processed_images=(step+1)*2,total_training_images=248,early_stopping=False,best_epoch=best_epoch,best_validation_Dice=best if best>=0 else None))
  js(out/'status.json',dict(state='validation',epoch=epoch,completed_epochs=epoch-1,maximum_epochs=100,early_stopping=False));m,vl,sub,per=evaluate(validation);tm=metrics(counts);scheduler.step(vl)
  row=dict(epoch=epoch,train_loss=loss_sum/248,validation_loss=vl,train_Dice=tm['Dice'],validation_Dice=m['Dice'],validation_IoU=m['IoU'],validation_Precision=m['Precision'],validation_Recall=m['Recall'],validation_Accuracy=m['Accuracy'],validation_Balanced_accuracy=m['Balanced_accuracy'],old_validation_Dice=sub['original_244']['Dice'],new_validation_Dice=sub['submitted_66']['Dice'],learning_rate=lr,training_views=248,optimizer_updates=updates,seconds=time.monotonic()-tic);history.append(row)
  if m['Dice']>best:
   best=m['Dice'];best_epoch=epoch;save(out/'weights/V35_310_best.pt',dict(state_dict=model.state_dict()));js(out/'selection.json',dict(best_epoch=best_epoch,validation_Dice=best,threshold=.5,checkpoint_SHA256=c.sha(out/'weights/V35_310_best.pt')))
  save(out/'weights/V35_310_epoch100.pt' if epoch==100 else out/'weights/V35_310_latest.pt',dict(state_dict=model.state_dict()))
  save(resume,dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,best_epoch=best_epoch,epoch=epoch,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),protocol_SHA256=c.sha(protocol_path)))
  csvout(out/'history.csv',history);js(out/'epoch_records'/('epoch_%03d.json'%epoch),dict(metrics=row,selected_best_epoch=best_epoch,complete_training_image_coverage=True));js(out/'status.json',dict(state='running' if epoch<100 else 'final_evaluation',completed_epochs=epoch,maximum_epochs=100,best_epoch=best_epoch,best_validation_Dice=best,early_stopping=False));print('EPOCH',epoch,json.dumps(row),'BEST',best_epoch,flush=True)
  if epoch%10==0:save(out/'milestones'/('epoch_%03d.pt'%epoch),dict(state_dict=model.state_dict()))
 final=[]
 for label,path in [('best_validation','V35_310_best.pt'),('epoch100','V35_310_epoch100.pt')]:
  model.load_state_dict(torch.load(out/'weights'/path,map_location='cpu',weights_only=True)['state_dict']);m,loss,sub,per=evaluate(validation,out/'validation_predictions'/label);final.append(dict(model=label,epoch=best_epoch if label=='best_validation' else 100,split='internal_validation',images=62,loss=loss,**m));csvout(out/(label+'_per_image.csv'),per)
  csvout(out/(label+'_by_source.csv'),[dict(source=s,**mm) for s,mm in sub.items()])
 csvout(out/'final_validation_metrics.csv',final);assert len(history)==101 and history[-1]['epoch']==100;assert all(r['training_views']==248 for r in history[1:]);assert c.sha(original)==original_hash and c.sha(HERE/'assets/weights/unet/best_model.pt')==original_hash
 js(out/'completion.json',dict(state='complete',epochs=100,early_stopping=False,images=310,train=248,val=62,test=0,independent_validation=False,selected_epoch=best_epoch,selected_validation_Dice=best,original_weights_preserved=True,original_weights_SHA256=original_hash,best_weights_SHA256=c.sha(out/'weights/V35_310_best.pt'),epoch100_weights_SHA256=c.sha(out/'weights/V35_310_epoch100.pt'),manifest_SHA256=c.sha(manifest)));js(out/'status.json',dict(state='complete',completed_epochs=100,best_epoch=best_epoch));print('COMPLETE_100_EPOCHS',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=20261124);main(p.parse_args())

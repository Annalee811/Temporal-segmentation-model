"""Fixed 100-epoch, full-coverage 250px V35 update with supplemental emphasis."""
import argparse, json, math, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
from models import RootUNet, objective
from train_merged_310 import js, save, csvout, metrics


def train(a):
    out=a.output.resolve(); source=a.source.resolve()
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    np.random.seed(a.seed);random.seed(a.seed)
    audit=c.read(out/'data_audit.json')
    assert c.sha(out/'data/manifest.csv')==audit['manifest_SHA256']
    assert c.sha(out/'data/parents.csv')==audit['parents_SHA256']
    rows=c.csvread(out/'data/manifest.csv');parents=c.csvread(out/'data/parents.csv')
    assert len(rows)==1240 and all(int(r['width'])==int(r['height'])==250 for r in rows)
    training=[r for r in rows if r['split']=='train']
    validation=[r for r in parents if r['split']=='val']
    assert len(training)==1048 and len(validation)==48
    assert sum(r['source']=='submitted_66' for r in training)==264
    assert all(r['source']=='original_244' for r in validation)
    assert not({r['group_id'] for r in training}&{r['group_id'] for r in validation})
    cache={}
    for r in rows:
        im=out/'data/RGB'/r['image_id'];gt=out/'data/mask'/r['image_id']
        assert c.sha(im)==r['RGB_sha256'] and c.sha(gt)==r['mask_sha256']
        cache[r['image_id']]=(np.array(Image.open(im).convert('RGB')),np.array(Image.open(gt)))
    parent_cache={}
    for r in validation:
        im=source/'data/RGB'/r['image_id'];gt=source/'data/mask'/r['image_id']
        assert c.sha(im)==r['RGB_sha256'] and c.sha(gt)==r['mask_sha256']
        parent_cache[r['image_id']]=(np.array(Image.open(im).convert('RGB')),np.array(Image.open(gt)))
    preserved={name:c.sha(out/'weights'/name) for name in audit['original_weights_preserved']}
    initial=out/'weights/V35_310_best.pt'
    model=RootUNet().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['state_dict'])
    mean_weight=sum(6 if r['source']=='submitted_66' else 1 for r in training)/len(training)
    protocol=dict(model='V35 ResNet18 U-Net',target='root vs background; all temporal roots merged',
        initialization='Warm start from latest V35_310_best.pt, epoch32 of the completed prior100-epoch run',
        initial_weights_SHA256=c.sha(initial),preserved_weights_SHA256=preserved,
        source_images=310,patches=1240,train_patches=1048,val_patches=192,
        train_parents=262,val_parents=48,test_parents=0,independent_test=False,
        all66_supplemental_parents_in_training=True,supplemental_training_patches=264,
        supplemental_validation_parents=0,
        epochs=100,early_stopping=False,seed=a.seed,physical_batch=a.batch,
        gradient_accumulation=max(1,8//a.batch),native_patch=250,further_random_cropping=False,
        full_training_patch_coverage_each_epoch=True,old_patch_weight=1,new_patch_weight=6,
        supplemental_fraction_of_weighted_patch_exposure=1584/2368,
        loss='Per-image original V35 hard-negative BCE + Dice + 0.15 clDice; supplemental images additionally 0.5*Tversky(alphaFP=0.3,betaFN=0.7); source weights old1,new6 normalized by epoch mean weight',
        lr_schedule='5-epoch linear warm-up 2e-5 to1e-4, cosine decay to2e-6 over fixed100epochs',
        weight_decay=1e-4,encoder_running_statistics='Warm-start BatchNorm running statistics held fixed; all convolution and affine weights trainable',
        augmentation='Full250px rotations, flips, brightness and gamma; no resized or random spatial crop',
        validation='Original parent grids reconstructed using250px Hann-weighted windows,stride188; each valid source pixel counted once',
        threshold=.5,checkpoint_selection='Maximum old-only internal-validation Dice; all66 supplemental images are training data, never used as validation; epoch100 separately retained',
        validation_caveat=audit['validation_caveat'],manifest_SHA256=audit['manifest_SHA256'],
        code_SHA256=c.sha(Path(__file__)),runtime=dict(torch=torch.__version__,threads=a.threads))
    if (out/'protocol.json').exists():assert c.read(out/'protocol.json')==protocol
    else:js(out/'protocol.json',protocol)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)

    def batch(rr):
        xx=[];yy=[];vv=[]
        for r in rr:
            im,gt=cache[r['image_id']];k=random.randrange(4)
            im=np.rot90(im,k);gt=np.rot90(gt,k)
            if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt)
            if random.random()<.5:im=np.flipud(im);gt=np.flipud(gt)
            im=np.clip(im.astype('float32')/255*random.uniform(.9,1.1),0,1)**random.uniform(.9,1.1)
            xx.append(im.copy());yy.append((gt==255).astype('float32'));vv.append((gt!=128).astype('float32'))
        return (torch.from_numpy(np.stack(xx).transpose(0,3,1,2).copy()).to(memory_format=torch.channels_last),
                torch.from_numpy(np.stack(yy)[:,None]),torch.from_numpy(np.stack(vv)[:,None]))

    def loss_value(z,y,v,rr):
        losses=[]
        for i,r in enumerate(rr):
            zi,yi,vi=z[i:i+1],y[i:i+1],v[i:i+1]
            loss=objective(zi,yi,vi)
            if r['source']=='submitted_66':
                p=zi.float().sigmoid()*vi;g=yi*vi;tp=(p*g).sum()
                fp=(p*(1-g)).sum();fn=((1-p)*g).sum()
                loss=loss+.5*(1-(tp+1)/(tp+.3*fp+.7*fn+1))
            losses.append(loss*(6 if r['source']=='submitted_66' else 1)/mean_weight)
        return torch.stack(losses).mean()

    @torch.inference_mode()
    def evaluate(folder=None):
        model.eval();counts=np.zeros(4,np.int64);sub=defaultdict(lambda:np.zeros(4,np.int64))
        per=[];losses=[];win=np.maximum(np.hanning(250)[:,None]*np.hanning(250)[None,:],.02).astype('float32')
        for r in validation:
            im,gt=parent_cache[r['image_id']];h,w=gt.shape
            xs=sorted(set(list(range(0,w-249,188))+[w-250]));ys=sorted(set(list(range(0,h-249,188))+[h-250]))
            coords=[(x,y) for y in ys for x in xs];sm=np.zeros((h,w),np.float32);wg=np.zeros_like(sm)
            for i in range(0,len(coords),a.batch):
                cc=coords[i:i+a.batch]
                x=torch.from_numpy(np.stack([im[y:y+250,x:x+250] for x,y in cc]).transpose(0,3,1,2).copy()).float().to(memory_format=torch.channels_last)/255
                y=torch.from_numpy(np.stack([(gt[yy:yy+250,xx:xx+250]==255).astype('float32') for xx,yy in cc])[:,None])
                v=torch.from_numpy(np.stack([(gt[yy:yy+250,xx:xx+250]!=128).astype('float32') for xx,yy in cc])[:,None])
                z=model(x);losses.append((float(objective(z,y,v)),len(cc)))
                for (xx,yy),p in zip(cc,z.sigmoid()[:,0].numpy()):
                    sm[yy:yy+250,xx:xx+250]+=p*win;wg[yy:yy+250,xx:xx+250]+=win
            assert (wg>0).all();p=sm/wg;cc=c.counts(gt,p>=.5);counts+=cc;sub[r['source']]+=cc
            per.append(dict(image_id=r['image_id'],source=r['source'],position=r['position'],period=r['period'],**metrics(cc)))
            if folder:
                folder.mkdir(parents=True,exist_ok=True);np.save(folder/(r['image_id']+'_probability.npy'),p)
                Image.fromarray((p>=.5).astype('uint8')*255).save(folder/(r['image_id']+'_mask.png'))
        return metrics(counts),{s:metrics(v) for s,v in sub.items()},per,sum(l*n for l,n in losses)/sum(n for _,n in losses)

    resume=out/'last_training_state.pt';history=[];best=-1;best_epoch=0;start=1
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['protocol_SHA256']==c.sha(out/'protocol.json')
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
        history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];start=ck['epoch']+1
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        print('RESUME',start,flush=True)
    else:
        m,sub,per,vl=evaluate(out/'baseline_predictions')
        csvout(out/'baseline_validation_per_image.csv',per)
        js(out/'baseline_validation.json',dict(overall=m,by_source=sub))
        history=[dict(epoch=0,train_loss=None,validation_loss=vl,train_Dice=None,
            validation_Dice=m['Dice'],validation_Recall=m['Recall'],validation_Precision=m['Precision'],validation_IoU=m['IoU'],
            old_validation_Dice=sub['original_244']['Dice'],old_validation_Recall=sub['original_244']['Recall'],
            old_train_Dice=None,new_train_Dice=None,old_train_Recall=None,new_train_Recall=None,
            selection_score=m['Dice'],
            learning_rate=2e-5,patches_visited=0,new_patches_visited=0,optimizer_updates=0,seconds=0)]
        csvout(out/'history.csv',history);print('BASELINE',json.dumps(history[0]),flush=True)
    accumulation=max(1,8//a.batch)
    for epoch in range(start,101):
        tic=time.monotonic();lr=(2e-5+8e-5*(epoch-1)/4) if epoch<=5 else 2e-6+(1e-4-2e-6)*.5*(1+math.cos(math.pi*(epoch-5)/95))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();model.encoder.eval();order=training.copy();random.shuffle(order)
        optimizer.zero_grad(set_to_none=True);loss_sum=0.;cc=np.zeros(4,np.int64);train_sub=defaultdict(lambda:np.zeros(4,np.int64));seen=set();steps=math.ceil(len(order)/a.batch);updates=0
        for step in range(steps):
            rr=order[step*a.batch:(step+1)*a.batch];x,y,v=batch(rr)
            with torch.autocast('cpu',dtype=torch.bfloat16):z=model(x)
            loss=loss_value(z,y,v,rr);assert torch.isfinite(loss)
            accumulation_size=min(accumulation,steps-(step//accumulation)*accumulation)
            (loss/accumulation_size).backward();loss_sum+=float(loss.detach())*len(rr)
            pp=z.detach().float().sigmoid()[:,0].numpy();gg=np.where(v[:,0].numpy()>.5,y[:,0].numpy()*255,128).astype('uint8')
            for g,p,r in zip(gg,pp,rr):
                sample_counts=c.counts(g,p>=.5);cc+=sample_counts;train_sub[r['source']]+=sample_counts
            seen.update(r['image_id'] for r in rr)
            if (step+1)%accumulation==0 or step+1==steps:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            if step%10==0:js(out/'status.json',dict(state='training',epoch=epoch,completed_epochs=epoch-1,total_epochs=100,patches_seen=len(seen),train_patches=1048,supplemental_training_patches=264,best_epoch=best_epoch,best_score=best if best>=0 else None))
        assert len(seen)==1048 and sum(r['source']=='submitted_66' for r in order)==264
        js(out/'status.json',dict(state='validation',epoch=epoch,completed_epochs=epoch-1,total_epochs=100))
        m,sub,per,vl=evaluate();score=m['Dice']
        row=dict(epoch=epoch,train_loss=loss_sum/1048,validation_loss=vl,train_Dice=metrics(cc)['Dice'],
            validation_Dice=m['Dice'],validation_Recall=m['Recall'],validation_Precision=m['Precision'],validation_IoU=m['IoU'],
            old_validation_Dice=sub['original_244']['Dice'],old_validation_Recall=sub['original_244']['Recall'],
            old_train_Dice=metrics(train_sub['original_244'])['Dice'],new_train_Dice=metrics(train_sub['submitted_66'])['Dice'],
            old_train_Recall=metrics(train_sub['original_244'])['Recall'],new_train_Recall=metrics(train_sub['submitted_66'])['Recall'],
            selection_score=score,learning_rate=lr,patches_visited=1048,new_patches_visited=264,optimizer_updates=updates,seconds=time.monotonic()-tic)
        history.append(row)
        if score>best:
            best=score;best_epoch=epoch;save(out/'weights/V35_full250_best.pt',dict(state_dict=model.state_dict()))
            js(out/'selection.json',dict(epoch=epoch,score=score,overall=m,by_source=sub,weights_SHA256=c.sha(out/'weights/V35_full250_best.pt')))
            csvout(out/'best_validation_per_image.csv',per)
        if epoch==100:save(out/'weights/V35_full250_epoch100.pt',dict(state_dict=model.state_dict()))
        save(resume,dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best,best_epoch=best_epoch,epoch=epoch,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),protocol_SHA256=c.sha(out/'protocol.json')))
        csvout(out/'history.csv',history);js(out/'epoch_records'/f'epoch_{epoch:03}.json',dict(metrics=row,full_1048_patch_coverage=True,all_264_supplemental_training_quadrants_used=True))
        print('EPOCH',epoch,json.dumps(row),'BEST',best_epoch,flush=True)
        if epoch%10==0:save(out/'milestones'/f'epoch_{epoch:03}.pt',dict(state_dict=model.state_dict()))
    final=[]
    for label,name in [('selected','V35_full250_best.pt'),('epoch100','V35_full250_epoch100.pt')]:
        model.load_state_dict(torch.load(out/'weights'/name,map_location='cpu',weights_only=True)['state_dict'])
        m,sub,per,vl=evaluate(out/'validation_predictions'/label)
        final.append(dict(model=label,epoch=best_epoch if label=='selected' else 100,**m))
        csvout(out/(label+'_validation_per_image.csv'),per)
        csvout(out/(label+'_validation_by_source.csv'),[dict(source=s,**mm) for s,mm in sub.items()])
    csvout(out/'final_validation_metrics.csv',final)
    assert len(history)==101 and all(r['patches_visited']==1048 and r['new_patches_visited']==264 for r in history[1:])
    assert all(c.sha(out/'weights'/name)==digest for name,digest in preserved.items())
    js(out/'completion.json',dict(state='complete',epochs=100,early_stopping=False,source_images=310,patches=1240,train_patches=1048,val_patches=192,
        supplemental_training_parents=66,supplemental_training_patches=264,supplemental_validation_parents=0,
        full_training_patch_coverage_every_epoch=True,source_pixels_discarded=0,selected_epoch=best_epoch,
        best_weights_SHA256=c.sha(out/'weights/V35_full250_best.pt'),epoch100_weights_SHA256=c.sha(out/'weights/V35_full250_epoch100.pt'),
        preserved_weights_SHA256=preserved,independent_test=False))
    js(out/'status.json',dict(state='complete',completed_epochs=100,total_epochs=100,best_epoch=best_epoch))
    print('COMPLETE_FULL_250_100_EPOCHS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=20261125)
    a=p.parse_args();assert a.batch in [2,4,8];train(a)

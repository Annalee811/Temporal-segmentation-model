"""One uninterrupted full-dataset training protocol, Python 3.5 compatible."""
from __future__ import division,print_function
import argparse,csv,json,math,os,random,time,traceback
from pathlib import Path
import numpy as np
import common as c


def js(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(str(tmp),str(path))


def csvout(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(str(tmp),str(path))


def tuples(value):return tuple(tuples(x) for x in value) if isinstance(value,list) else value


def rng():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[1].tolist(),n[2],n[3],n[4]])


def restore_rng(state):
    random.setstate(tuples(state['python']));n=state['numpy']
    np.random.set_state(('MT19937',np.array(n[0],np.uint32),n[1],n[2],n[3]))


def run(args,Backend):
    root=args.run;out=root/args.model;out.mkdir(exist_ok=True)
    if (out/'completion.json').exists():print('ALREADY_COMPLETE',args.model,flush=True);return
    protocol=c.read(root/'protocol.json');seed=protocol['seed']
    random.seed(seed);np.random.seed(seed)
    data=c.Data(root);training=[r for r in data.rows if r['split']=='train'];validation=[r for r in data.rows if r['split']=='val']
    assert len(training)==200 and len(validation)==20
    js(out/'status.json',dict(state='initializing',model=args.model,completed_epochs=0,maximum_epochs=protocol['maximum_epochs']))
    backend=Backend(args,protocol)
    initial=out/('initial_model'+backend.suffix);best=out/('best_model'+backend.suffix);last=out/('last'+backend.resume_suffix)
    history=[]
    def status(phase,epoch,processed=0,total=0):
        js(out/'status.json',dict(state='running',model=args.model,phase=phase,epoch=epoch,
           completed_epochs=history[-1]['epoch'] if history else 0,
           processed_views=processed,total_views=total,maximum_epochs=protocol['maximum_epochs'],updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    def evaluate(rows,epoch,label):
        total=np.zeros(4,np.int64);loss_sum=0.;n=0
        for i in range(0,len(rows),2):
            rr=rows[i:i+2];xx,yy,vv=data.batch(rr,False);pp,loss=backend.evaluate(xx,yy,vv)
            assert np.isfinite(loss) and np.isfinite(pp).all()
            loss_sum+=float(loss)*len(rr);n+=len(rr)
            for row,p in zip(rr,pp):
                _,target=data.load(row);total+=c.counts(target,p>=.5)
            if i%20==0:status(label,epoch,n,len(rows))
        m=c.metrics(total);m['Accuracy']=float((total[0]+total[3])/total.sum())
        return loss_sum/n,m
    def record(epoch,train_opt,train_eval,validation_loss,train_m,val_m,lr,seconds):
        result=dict(epoch=epoch,train_optimization_loss=train_opt,train_eval_loss=train_eval,
           validation_loss=validation_loss,learning_rate_used=lr,learning_rate_next=control['lr'],
           validation_Dice=val_m['Dice'],validation_IoU=val_m['IoU'],validation_Precision=val_m['Precision'],
           validation_Recall=val_m['Recall'],validation_Accuracy=val_m['Accuracy'],
           validation_TP=val_m['TP'],validation_FP=val_m['FP'],validation_FN=val_m['FN'],validation_TN=val_m['TN'],
           train_eval_Dice=train_m['Dice'],train_eval_IoU=train_m['IoU'],threshold=.5,
           training_views_visited=0 if epoch==0 else 200,optimizer_updates_this_epoch=0 if epoch==0 else 25,
           seconds=seconds)
        return result
    if last.exists():
        meta=backend.restore(last)
        current_protocol=c.sha(root/'protocol.json')
        if meta['protocol_SHA256']!=current_protocol:
            assert meta['protocol_SHA256']==protocol['parent_protocol_SHA256'], 'Unexpected parent protocol'
            js(out/'continuation_origin.json',dict(parent_protocol_SHA256=meta['protocol_SHA256'],
                parent_completed_epoch=meta['completed_epoch'],parent_current_epoch=meta['current_epoch'],
                parent_cursor=meta['cursor'],parent_optimizer_updates=meta['optimizer_updates'],
                parent_stop_flag=meta['control']['stop'],resume_checkpoint_SHA256=c.sha(last),
                change='Fixed 100-epoch budget; clear previous early-stop flag; preserve optimizer/RNG/order/scheduler'))
            meta['protocol_SHA256']=current_protocol
            meta['control']['stop']=False
        assert meta['protocol_SHA256']==current_protocol
        restore_rng(meta['rng']);control=meta['control'];history=meta['history']
        print('RESUME_SAME_TRAJECTORY',args.model,meta['completed_epoch'],meta['current_epoch'],meta['cursor'],flush=True)
    else:
        backend.save_model(initial);initial_hash=c.sha(initial)
        control=dict(lr=protocol['initial_lr'],best_dice=-1.,best_epoch=0,early_anchor=-1.,early_bad=0,
                     scheduler_best=None,scheduler_bad=0,stop=False)
        te,tm=evaluate(training,0,'initial_train_evaluation');vl,vm=evaluate(validation,0,'initial_validation')
        control.update(best_dice=vm['Dice'],early_anchor=vm['Dice'],scheduler_best=vl)
        backend.save_model(best)
        js(out/'best.json',dict(epoch=0,validation_Dice=vm['Dice'],threshold=.5,model_SHA256=c.sha(best)))
        history=[record(0,None,te,vl,tm,vm,control['lr'],0.)]
        meta=dict(protocol_SHA256=c.sha(root/'protocol.json'),initial_model_SHA256=initial_hash,
          completed_epoch=0,current_epoch=1,cursor=0,order=[],loss_sum=0.,seen=0,optimizer_updates=0,
          control=control,history=history,rng=rng())
        backend.save_resume(last,meta);csvout(out/'history.csv',history)
        js(out/'initialization.json',dict(initialization='fresh random parameters; no task-trained or external weights loaded',
           seed=seed,initial_model_SHA256=initial_hash,source_model_code_SHA256=c.sha(Path(__file__).with_name('models.py')),
           current_reference_manifest_SHA256=c.sha(root/'data/manifest.csv')))
        print('INITIAL_EVALUATION_SAVED',args.model,'train',te,'val',vl,'Dice',vm['Dice'],flush=True)
    def checkpoint():
        meta['control']=control;meta['history']=history;meta['rng']=rng();backend.save_resume(last,meta)
    for epoch in range(meta['current_epoch'],protocol['maximum_epochs']+1):
        if control['stop']:break
        tic=time.time();lr=control['lr'];backend.set_lr(lr)
        if not meta['order']:
            order=list(range(len(training)));random.Random(seed+epoch).shuffle(order)
            meta.update(current_epoch=epoch,order=order,cursor=0,loss_sum=0.,seen=0)
            csvout(out/'orders'/('epoch_%04d.csv'%epoch),[dict(index=i,image_id=training[k]['image_id']) for i,k in enumerate(order)])
        assert sorted(meta['order'])==list(range(200)) and meta['cursor']%8==0
        for cursor in range(meta['cursor'],200,2):
            rr=[training[k] for k in meta['order'][cursor:cursor+2]];xx,yy,vv=data.batch(rr,True)
            loss=backend.train_batch(xx,yy,vv);assert np.isfinite(loss),'Nonfinite optimization loss'
            meta['loss_sum']+=float(loss)*len(rr);meta['seen']+=len(rr);meta['cursor']=cursor+2
            if (cursor+2)%8==0:
                backend.step();meta['optimizer_updates']+=1;status('optimization',epoch,cursor+2,200)
                if meta['optimizer_updates']%10==0 or cursor+2==200:checkpoint()
        assert meta['seen']==200 and meta['cursor']==200
        te,tm=evaluate(training,epoch,'fixed_train_evaluation');vl,vm=evaluate(validation,epoch,'validation')
        if control['best_epoch']==0 or vm['Dice']>control['best_dice']:
            control['best_dice']=vm['Dice'];control['best_epoch']=epoch;backend.save_model(best)
            js(out/'best.json',dict(epoch=epoch,validation_Dice=vm['Dice'],threshold=.5,model_SHA256=c.sha(best)))
        if vm['Dice']>control['early_anchor']+protocol['early_stopping_min_delta']:
            control['early_anchor']=vm['Dice'];control['early_bad']=0
        else:control['early_bad']+=1
        if vl<control['scheduler_best']-1e-4:
            control['scheduler_best']=vl;control['scheduler_bad']=0
        else:control['scheduler_bad']+=1
        if control['scheduler_bad']>=protocol['lr_patience']:
            control['lr']=max(protocol['minimum_lr'],control['lr']*protocol['lr_factor']);control['scheduler_bad']=0
        if epoch>=protocol['minimum_epochs'] and control['early_bad']>=protocol['early_stopping_patience']:
            control['stop']=True
        row=record(epoch,meta['loss_sum']/meta['seen'],te,vl,tm,vm,lr,time.time()-tic);history.append(row)
        meta.update(completed_epoch=epoch,current_epoch=epoch+1,cursor=0,order=[],loss_sum=0.,seen=0)
        checkpoint();csvout(out/'history.csv',history)
        js(out/'epoch_records'/('epoch_%04d.json'%epoch),dict(metrics=row,last_checkpoint_SHA256=c.sha(last),
            complete_dataset_coverage=True,unique_training_views=200,selected_best_epoch=control['best_epoch']))
        if epoch%10==0:backend.save_model(out/'milestones'/('epoch_%04d'%epoch+backend.suffix))
        print('FULL_EPOCH',args.model,epoch,'optimization_loss',row['train_optimization_loss'],
              'train_eval',te,'val_loss',vl,'val_Dice',vm['Dice'],'seconds',round(row['seconds'],1),flush=True)

    js(out/'status.json',dict(state='evaluating_final_model',completed_epochs=meta['completed_epoch'],selected_epoch=control['best_epoch']))
    backend.load_model(best);final=[]
    for split in ['val','test']:
        rows=[r for r in data.rows if r['split']==split];total=np.zeros(4,np.int64);per=[]
        folder=out/'probabilities'/split;folder.mkdir(parents=True,exist_ok=True)
        for i in range(0,len(rows),2):
            rr=rows[i:i+2];xx,_,_=data.batch(rr,False);pp=backend.predict(xx)
            for row,p in zip(rr,pp):
                assert np.isfinite(p).all();np.save(str(folder/(row['image_id']+'_probability.npy')),p.astype(np.float32))
                _,target=data.load(row);cc=c.counts(target,p>=.5);total+=cc
                per.append(dict(image_id=row['image_id'],tile_id=row['tile_id'],period=row['period'],**c.metrics(cc)))
        csvout(out/(split+'_per_tile_metrics.csv'),per)
        final.append(dict(model=args.model,split=split,threshold=.5,views=len(rows),**c.metrics(total)))
    csvout(out/'final_metrics.csv',final)
    assert c.sha(initial)==meta['initial_model_SHA256']
    js(out/'completion.json',dict(state='complete',epochs=meta['completed_epoch'],selected_epoch=control['best_epoch'],
       early_stopped=control['stop'],checkpoint_reloads_during_training=0,
       data_reference='released binary annotation reference',initial_model_SHA256=c.sha(initial),best_model_SHA256=c.sha(best),
       final_evaluation_threshold=.5,test_evaluated_only_after_selection=True))
    js(out/'status.json',dict(state='complete',completed_epochs=meta['completed_epoch'],selected_epoch=control['best_epoch']))
    backend.close();print('CONTINUOUS_TRAINING_COMPLETE',args.model,flush=True)


def entry(Backend):
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--model',choices=['unet','segformer','deeplab'],required=True)
    parser.add_argument('--threads',type=int,default=1);args=parser.parse_args()
    try:run(args,Backend)
    except Exception as error:
        js(args.run/args.model/'error.json',dict(state='error',exception=repr(error),traceback=traceback.format_exc()))
        js(args.run/args.model/'status.json',dict(state='error',exception=repr(error)))
        raise

from paths import ROOT,WORK
"""Close EVERY inferred mixed-axis state and suppress third classes at junctions."""

import json

import pickle

import sys

from pathlib import Path

import numpy as np

from PIL import Image

from scipy import ndimage as ndi

OUT = WORK/'junction'

PREV = WORK/'axis'

REF = WORK/'reference'

PERIODS=['T1','T2','T3','T4']

from engine import review_setup

from paths import js,csvout,sha

from decode import runs

def logit(p):
    p=float(np.clip(p,.01,.99));return float(np.log(p/(1-p)))

def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    OUT.mkdir(exist_ok=True)
    for name in ['full','changes','ownership']: (OUT/name).mkdir(exist_ok=True)
    with (PREV/'root_axis_snapshots.pkl').open('rb') as f:snap=pickle.load(f)
    inputs=[PREV/'full'/(t+'_classes.png') for t in PERIODS]+[ROOT/'joint/joint_best.pt',REF/'reference_provenance.json']
    immutable={str(p):sha(p) for p in inputs}
    setup=review_setup();allrows=[];events=[];pixelrows=[];modes_by_period={};fragment_rows=[];radii_rows=[]
    for ti,t in enumerate(PERIODS):
        old=np.array(Image.open(PREV/'full'/(t+'_classes.png')));out=old.copy();mask=old>0
        owner=np.load(PREV/'ownership'/(t+'_axis_owner.npz'))['axis'].copy()
        tp=setup['top']['topos'][t];points=snap[t]['points'];modes={};records={}
        if ti==0:
            Image.fromarray(out).save(OUT/'full'/(t+'_classes.png'));Image.fromarray(np.zeros(old.shape,np.uint8)).save(OUT/'changes'/(t+'_edited.png'));np.savez_compressed(OUT/'ownership'/(t+'_axis_owner.npz'),axis=owner);pixelrows.append(dict(period=t,changed_pixels=0,root_pixels_added=0,root_pixels_removed=0));continue


        for r in snap[t]['states']:
            ai=r['axis'];use=owner==ai;v=old[use];counts=np.bincount(v,minlength=5);p=points[ai]
            mixed=bool(counts[2] and (counts[1] or counts[3]));state=r['state'];score=None;past=0.;coverage=0.;affinity=0.;confidence=1.;decision='inherited accepted state'
            if state not in ['new','established']:
                if not counts[2]:state='established';decision='already old-compatible single-axis classes; no body repaint'
                elif not(counts[1] or counts[3]):state='new';decision='already new-only axis; no body repaint'
                else:
                    history=setup['history'](ti,p);past=max(history[:ti] or [0.]);coverage,affinity=setup['track_evidence'](ti,p)


                    score=logit(r['learned_new_probability'])+.65*np.log((counts[2]+2)/(counts[1]+2))-3*past
                    if affinity>=.55:score-=1.5*max(0.,coverage-.5)
                    positive_old=(r['prior_support']>=8 and r['identity_affinity']>=setup['identity_threshold']) or (coverage>=.7 and affinity>=.65 and (past>=.15 or counts[1]>=10))
                    if positive_old:state='established';decision='previous same-root support overrides a second birth'
                    else:state='new' if score>=0 else 'established';decision='MAP state from learned root-state head, shaft evidence and multi-period correspondence'
                    confidence=float(1/(1+np.exp(-abs(float(score)))))
            modes[ai]=state
            if mixed:
                if state=='new':out[use]=2
                else:



                    if not counts[3]:out[use&(old==2)]=1
                    elif not counts[1]:out[use&(old==2)]=3
                    else:
                        vv=old[p[:,1],p[:,0]];purple=np.flatnonzero(vv==1);yellow=np.flatnonzero(vv==3)
                        new_positions=np.flatnonzero(vv==2);target=vv.copy()
                        for k in new_positions:
                            left=np.flatnonzero(vv[:k]==1);right=np.flatnonzero(vv[k+1:]==1)
                            bounded=bool(len(left) and len(right))
                            dp=float(np.min(np.abs(purple-k))) if len(purple) else np.inf;dy=float(np.min(np.abs(yellow-k))) if len(yellow) else np.inf
                            target[k]=1 if bounded or dp<=dy else 3


                        from scipy.spatial import cKDTree
                        yy,xx=np.where(use&(old==2));_,near=cKDTree(p).query(np.c_[xx,yy]);chosen=target[near];bad=chosen==2
                        if bad.any():
                            good=np.flatnonzero(np.isin(target,[1,3]));_,nn=cKDTree(p[good]).query(np.c_[xx[bad],yy[bad]]);chosen[bad]=target[good][nn]
                        out[yy,xx]=chosen
            records[ai]=dict(period=t,axis=ai,original_state=r['state'],final_state=state,was_mixed=mixed,previous_persistent_pixels=int(counts[1]),previous_new_pixels=int(counts[2]),previous_extension_pixels=int(counts[3]),learned_new_probability=r['learned_new_probability'],decision_score=None if score is None else float(score),decision_confidence=confidence,low_confidence=bool(mixed and confidence<.75),past_direct_fraction=past,flexible_prior_coverage=coverage,flexible_prior_affinity=affinity,decision=decision,x_start=r['x_start'],y_start=r['y_start'],x_end=r['x_end'],y_end=r['y_end'])
        modes_by_period[t]=modes




        from scipy.spatial import cKDTree
        new_points=[];new_ids=[];old_points=[]
        for ai,p in enumerate(points):
            if modes[ai]=='new':new_points.extend(p);new_ids.extend([ai]*len(p))
            elif len(p)>=12:old_points.extend(p)
        new_tree=cKDTree(np.asarray(new_points)) if new_points else None;old_tree=cKDTree(np.asarray(old_points)) if old_points else None
        new_ids=np.asarray(new_ids,np.int32);canonical={}
        if new_tree is not None:
            for ai,rec in records.items():
                if snap[t]['states'][ai]['state']!='uncertain' or len(points[ai])>8 or rec['learned_new_probability']<.5:continue
                use=owner==ai
                if use.sum()>32 or not (use&np.isin(out,[1,3])).any():continue
                yy,xx=np.where(use);xy=np.c_[xx,yy];dist,nn=new_tree.query(xy);other=old_tree.query(xy)[0] if old_tree is not None else np.full(len(xy),np.inf)
                if dist.max()<=8 and np.median(other-dist)>=2:
                    vals,cnt=np.unique(new_ids[nn],return_counts=True);parent=int(vals[cnt.argmax()]);owner[use]=parent;out[use]=2;canonical[ai]=parent
                    fragment_rows.append(dict(period=t,fragment_axis=ai,joined_new_axis=parent,pixels=int(use.sum()),maximum_gap=float(dist.max()),median_margin_to_old_axis=float(np.median(other-dist)),learned_new_probability=rec['learned_new_probability'],reason='tiny disconnected piece uniquely matches nearby new axis; foreground is not filled'))


        core=np.zeros(old.shape,np.uint8);footprint=np.zeros(old.shape,bool);native_half_width=ndi.distance_transform_edt(mask)
        for hubid,hub in enumerate(tp['hubs']):
            if hub.get('disabled') or len(hub['arms'])<3:continue
            p=hub['points'];core[p[:,1],p[:,0]]=1
            radius=int(np.clip(np.ceil(4*np.median(native_half_width[p[:,1],p[:,0]])),4,10));radii_rows.append(dict(period=t,hub=hubid,native_neighborhood_radius=radius))
            x0=max(0,int(p[:,0].min())-radius);x1=min(old.shape[1],int(p[:,0].max())+radius+1);y0=max(0,int(p[:,1].min())-radius);y1=min(old.shape[0],int(p[:,1].max())+radius+1)
            local=np.ones((y1-y0,x1-x0),bool);local[p[:,1]-y0,p[:,0]-x0]=False;near=ndi.distance_transform_edt(local)<=radius;footprint[y0:y1,x0:x1]|=near&mask[y0:y1,x0:x1]
        Image.fromarray(footprint.astype(np.uint8)*255).save(OUT/'changes'/(t+'_junction_footprint.png'))
        groups,n=ndi.label(footprint,np.ones((3,3)))
        event_hubs={}
        for j,hub in enumerate(tp['hubs']):
            if hub.get('disabled') or len(hub['arms'])<3:continue
            p=hub['points'];ids=groups[p[:,1],p[:,0]];ids=ids[ids>0]
            if len(ids):event_hubs.setdefault(int(np.bincount(ids).argmax()),[]).append(j)
        rootof=snap[t]['routes']['rootof']
        for event,hubids in event_hubs.items():
            axes=set();armtypes=[];axis_arm_types={}
            for j in hubids:
                for arm in tp['hubs'][j]['arms']:
                    ai=int(rootof[arm['segment']]);ai=canonical.get(ai,ai);axes.add(ai);p=tp['segments'][arm['segment']]['points'];p=p if arm['end']==0 else p[::-1];p=p[:32];p=p[groups[p[:,1],p[:,0]]==0]
                    if modes[ai]=='new':armtypes.append(2);axis_arm_types.setdefault(ai,set()).add(2);continue
                    vals=out[p[:,1],p[:,0]] if len(p) else np.array([],np.uint8);vals=vals[np.isin(vals,[1,3])]
                    if len(vals):c=1 if (vals==1).sum()>=(vals==3).sum() else 3
                    else:
                        rec=records[ai];c=1 if rec['previous_persistent_pixels']>=rec['previous_extension_pixels'] else 3
                    armtypes.append(c);axis_arm_types.setdefault(ai,set()).add(c)



            region=groups==event;yy_event,xx_event=np.where(region);center=np.array([np.median(xx_event),np.median(yy_event)])
            for ai in np.unique(owner[region]):
                ai=int(ai)
                if ai<0 or ai in axes:continue
                axes.add(ai)
                if modes[ai]=='new':c=2
                else:
                    q=points[ai];q=q[groups[q[:,1],q[:,0]]==0]
                    if len(q):
                        q=q[np.argsort(np.linalg.norm(q-center,axis=1))[:8]];vv=out[q[:,1],q[:,0]];c=1 if (vv==1).sum()>=(vv==3).sum() else 3
                    else:
                        r=records[ai];c=1 if r['previous_persistent_pixels']>=r['previous_extension_pixels'] else 3
                armtypes.append(c);axis_arm_types.setdefault(ai,set()).add(c)
            axes=sorted(axes);types=set(armtypes)


            if 1 in types and 2 in types:allowed={1,2};rule='new/persistent event: no extension in the shared footprint'
            elif 2 in types and 3 in types:allowed={2,3};rule='new/extension event: no persistent third class'
            elif types=={2}:allowed={2};rule='new/new event'
            elif types=={1}:allowed={1};rule='persistent/persistent event'
            elif types=={3}:allowed={3};rule='extension/extension event'
            else:allowed={1,3};rule='old-root event: persistent and/or extension only'
            use=(groups==event)&np.isin(owner,axes);yy,xx=np.where(use)
            before_count=int((use&~np.isin(old,list(allowed))).sum())
            ownership_changed=0
            for ai in axes:
                bad=use&(owner==ai)&~np.isin(out,list(allowed))
                if not bad.any():continue
                target=2 if modes[ai]=='new' else (1 if 1 in allowed else 3)
                assert target in allowed
                eligible=[k for k in axes if target in axis_arm_types.get(k,set()) and(modes[k]=='new')==(target==2)]
                if eligible and ai not in eligible:


                    from scipy.spatial import cKDTree
                    yy0,xx0=np.where(bad);xy0=np.c_[xx0,yy0];distances=np.stack([cKDTree(points[k]).query(xy0)[0] for k in eligible]);chosen=np.array(eligible)[distances.argmin(0)];ownership_changed+=int((owner[yy0,xx0]!=chosen).sum());owner[yy0,xx0]=chosen
                out[bad]=target
            assert not (use&~np.isin(out,list(allowed))).any()
            events.append(dict(period=t,event=event,hubs=','.join(map(str,hubids)),incident_axes=','.join(map(str,axes)),stable_arm_types=','.join(map(str,sorted(types))),allowed_types=','.join(map(str,sorted(allowed))),pixels=int(use.sum()),forbidden_before=before_count,forbidden_after=0,ownership_reassigned_pixels=ownership_changed,rule=rule,x=int(np.median(xx)),y=int(np.median(yy))))
        for ai,rec in records.items():
            v=out[owner==ai];bad=int((v!=2).sum()) if rec['final_state']=='new' else int((v==2).sum());assert bad==0
            rec['forbidden_after']=bad;rec['new_extension_after']=bool((v==2).any() and(v==3).any());rec['new_persistent_after']=bool((v==2).any() and(v==1).any());rec['joined_parent_axis']=canonical.get(ai,-1);allrows.append(rec)
        assert np.array_equal(out>0,mask)
        Image.fromarray(out).save(OUT/'full'/(t+'_classes.png'));Image.fromarray((old!=out).astype(np.uint8)*255).save(OUT/'changes'/(t+'_edited.png'));np.savez_compressed(OUT/'ownership'/(t+'_axis_owner.npz'),axis=owner)
        pixelrows.append(dict(period=t,changed_pixels=int((old!=out).sum()),root_pixels_added=0,root_pixels_removed=0));print('ALL_AXIS_REPAIR',t,'mixed_checked',sum(r['was_mixed'] for r in records.values()),'changes',int((old!=out).sum()),'junction_events',len(event_hubs),flush=True)
    for p,digest in immutable.items():assert sha(p)==digest
    csvout(OUT/'Every_axis_state_audit.csv',allrows);csvout(OUT/'Every_junction_event_audit.csv',events);csvout(OUT/'Native_pixel_changes.csv',pixelrows)
    if fragment_rows:csvout(OUT/'Disconnected_fragment_identity_corrections.csv',fragment_rows)
    csvout(OUT/'Native_width_junction_neighborhoods.csv',radii_rows)
    mixed=[r for r in allrows if r['was_mixed']];csvout(OUT/'Previously_unresolved_mixed_axes_now_processed.csv',mixed);low=[r for r in mixed if r['low_confidence']]
    if low:csvout(OUT/'Low_confidence_state_decisions.csv',low)
    js(OUT/'consistency_verification.json',dict(passed=True,T1_baseline_axes_checked=len(snap['T1']['points']),total_four_period_axes_checked=len(allrows)+len(snap['T1']['points']),all_inferred_temporal_axes_scanned=len(allrows),mixed_axes_before=len(mixed),mixed_new_extension_before=sum(bool(r['previous_new_pixels'] and r['previous_extension_pixels']) for r in mixed),mixed_axes_after=0,new_extension_after=0,new_persistent_after=0,all_junction_events_scanned=len(events),forbidden_event_pixels_before=sum(r['forbidden_before'] for r in events),forbidden_event_pixels_after=0,ownership_reassigned_junction_pixels=sum(r['ownership_reassigned_pixels'] for r in events),disconnected_new_root_fragments_rejoined=len(fragment_rows),junction_radius_policy='two local native diameters, clamped to4..10px; foreground itself unchanged',low_confidence_mixed_decisions=len(low),geometry_unchanged=True,reference_unchanged=True,no_training=True,immutable_input_sha256=immutable,prediction_sha256={t:sha(OUT/'full'/(t+'_classes.png')) for t in PERIODS},scope='Consistency of inferred axes and event footprints, NOT independent biological accuracy; MAP decisions on low-confidence axes are explicitly exported.'))

if __name__=='__main__':main()

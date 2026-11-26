"""Matched whole-pipeline inference ablation with current V35 foreground.
Compare frozen V66 priors with the complete deployed TRGD transfer pipeline.
No model is retrained; no individual TRGD component is ablated separately.
"""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/workspace/Temporal-segmentation-model');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));from evaluate import merged_metrics,REFERENCE
FULL=ROOT/'work/V90_all_matched_native';RUNTIME=ROOT/'work/V90_full250_common66';V35=ROOT/'work/V35_full250_common66'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arr(p):return np.array(Image.open(p))
def writecsv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def measure(gt,p):
 use=np.isin(gt,[1,2,3]);y=(gt[use]==2).astype(np.int64);z=np.where(p[use]==0,2,p[use]-1);return np.bincount(y*3+z,minlength=6).reshape(2,3)
def main():
 rows=list(csv.DictReader(open(FULL/'prediction_manifest.csv',encoding='utf-8-sig')));assert len(rows)==66
 provenance=[];per=[];cm={'off':np.zeros((2,3),np.int64),'on':np.zeros((2,3),np.int64)};cm_conditional={k:np.zeros((2,3),np.int64) for k in cm}
 reference_position='0353-2-B3';reference_pixels=0;reference_covered=0
 for i,r in enumerate(rows,1):
  pos,t=r['position'],r['period'];folder=FULL/'predictions'/pos;priorpath=RUNTIME/'runs'/pos/'priors'/pos/(t+'_prior_classes.png');fgpath=folder/(t+'_V35_mask.png');fullpath=folder/(t+'_V90_classes.png');rgbpath=folder/(t+'_RGB.png')
  assert sha(fgpath)==r['V35_mask_SHA256'] and sha(fullpath)==r['V90_classes_SHA256'] and sha(rgbpath)==r['RGB_SHA256']
  mask=arr(fgpath)>0;prior=arr(priorpath);on=arr(fullpath);off=np.where(mask,np.where(prior==2,2,1),0).astype('uint8')
  assert np.isin(prior,[0,1,2,3,4]).all() and np.array_equal(prior>0,mask)
  assert np.isin(on,[0,1,2]).all() and np.array_equal(on>0,mask)
  assert np.array_equal(off>0,on>0) and off.shape==(int(r['height']),int(r['width']))
  if t=='T1':assert not (off==2).any() and not (on==2).any()
  dest=HERE/'trgd_off'/pos;dest.mkdir(parents=True,exist_ok=True);out=dest/(t+'_classes.png');Image.fromarray(off).save(out)
  provenance.append(dict(position=pos,period=t,RGB_SHA256=r['RGB_SHA256'],V35_mask_SHA256=r['V35_mask_SHA256'],TRGD_off_SHA256=sha(out),TRGD_on_SHA256=r['V90_classes_SHA256'],root_pixels=int(mask.sum()),off_new_root_pixels=int((off==2).sum()),on_new_root_pixels=int((on==2).sum()),foreground_difference_pixels=0,changed_root_state_pixels=int((off!=on).sum()),included_in_accuracy_evaluation=pos==reference_position and t!='T1',TRGD_off_source=str(priorpath),TRGD_on_source=str(fullpath)))
  if pos==reference_position:
   native=arr(ROOT/'03_temporal_V66/assets/RGB'/(t+'_RGB.png'));same=arr(rgbpath);assert np.array_equal(native,same),'Reference RGB pixels differ'
   if t!='T1':
    gt=arr(REFERENCE/(t+'_classes.png'));use=np.isin(gt,[1,2,3]);reference_pixels+=int(use.sum());reference_covered+=int((use&mask).sum())
    for key,pred in [('off',off),('on',on)]:
     matrix=measure(gt,pred);cm[key]+=matrix;per.append(dict(model='TRGD '+key,period=t,**merged_metrics(matrix)))
     conditional=gt.copy();conditional[~mask]=0;cm_conditional[key]+=measure(conditional,pred)
  if i%12==0:print('VERIFIED',i,'of66',flush=True)
 assert reference_pixels==468203
 result=[dict(model='TRGD '+k,scope='annotated_reference_roots_including_missed',**merged_metrics(cm[k])) for k in ['off','on']]
 cond=[dict(model='TRGD '+k,scope='reference_roots_detected_by_fixed_V35',**merged_metrics(cm_conditional[k])) for k in ['off','on']]
 writecsv(HERE/'metrics.csv',result);writecsv(HERE/'per_period_metrics.csv',per);writecsv(HERE/'conditional_metrics.csv',cond);writecsv(HERE/'paired_prediction_audit.csv',provenance)
 proof=dict(experiment='Whole-pipeline inference ablation: TRGD off versus TRGD on',binary_model='V35 selected epoch91 after100epochs',binary_weights_SHA256=rows[0]['binary_weights_SHA256'],temporal_weights_SHA256=rows[0]['V90_weights_SHA256'],weights_retrained=False,TRGD_off='Frozen V66 priors; no V90 ConvLSTM/GAT or Axis/Junction decoding',TRGD_on='Frozen V90 ConvLSTM+GAT followed by Axis and Junction decoding in the existing deployed transfer interface',paired_positions=18,paired_images=66,all_foreground_difference_pixels=0,reference_position=reference_position,evaluated_periods=['T2','T3','T4'],T1='Context only; not scored',reference_root_pixels=reference_pixels,reference_root_pixels_detected_by_V35=reference_covered,V35_reference_root_coverage=reference_covered/reference_pixels,evaluation_scope='One previously annotated development sequence; no independent temporal test set. Other positions have no complete verified two-class temporal reference used here.',class_codes={'0':'background','1':'existing_root','2':'new_root'},reference_merge='legacy codes1and3 merged to existing; code2 new',metrics=result,conditional_metrics=cond,confusion={k:cm[k].tolist() for k in cm},conditional_confusion={k:cm_conditional[k].tolist() for k in cm_conditional},source_reference_SHA256={t:sha(REFERENCE/(t+'_classes.png')) for t in ['T2','T3','T4']},transfer_adapter_known_departures=json.loads((FULL/'completion.json').read_text())['known_preprocessing_departures'])
 (HERE/'evaluation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2),flush=True)
if __name__=='__main__':main()

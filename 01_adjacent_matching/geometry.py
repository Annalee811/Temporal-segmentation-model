"""Reproducible original RGB, regional crops, exact area and cumulative RoMa QC.

Run geometry first. This module never applies dense/nonrigid fields to RGB.
Polygon unions are measured in each original's native pixel-area coordinates.
"""
import argparse,csv,hashlib,io,json,math,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/registration_matplotlib')
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from shapely.geometry import Polygon,box as sbox
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
COLORS=['#32A8BF','#D39439','#9971B4','#73A86A']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def proj(p,H):return (p@np.asarray(H)[:2,:2].T+np.asarray(H)[:2,2]).astype('float32')
def grid(box):
    x0,y0,x1,y1=box;y,x=np.mgrid[y0:y1,x0:x1].astype('float32');return np.stack([x,y],-1)
def sample(a,p,wh):
    h,w=a.shape[:2];q=(p+.5)*[w/wh[0],h/wh[1]]-.5
    return cv2.remap(a.astype('float32'),q[...,0].astype('float32'),q[...,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
def corners(b,edges=False):
    x0,y0,x1,y1=b
    if edges:return np.array([[x0-.5,y0-.5],[x1-.5,y0-.5],[x1-.5,y1-.5],[x0-.5,y1-.5]])
    return np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32)
def join(parts,ids,width,gap=24,value=255):
    arrays=[]
    for i,k in enumerate(ids):
        if i:arrays.append(np.full((gap,width)+parts[k].shape[2:],value,dtype=parts[k].dtype))
        arrays.append(parts[k])
    return np.concatenate(arrays,axis=0)
def csvwrite(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def native_mask(polys,wh):

    w,h=wh;m=np.zeros((h,w),np.uint8)
    for poly in polys:cv2.fillConvexPoly(m,np.rint(poly*16).astype('int32'),255,shift=4)
    return m

def render_position(work,entry,force=False):
    pos=entry['position'];raw=work/'inputs/raw'/pos;out=work/'deliverable'/pos;out.mkdir(parents=True,exist_ok=True)
    if (out/'geometry_metadata.json').exists() and not force:return json.loads((out/'geometry_metadata.json').read_text())
    paths={int(p.stem[1:]):p for p in raw.iterdir() if p.is_file()};periods=sorted(paths);images={t:np.array(Image.open(p).convert('RGB')) for t,p in paths.items()};wh={t:im.shape[1::-1] for t,im in images.items()}
    regions=entry['regions'];ids=[r['id'] for r in regions];byid={r['id']:r for r in regions};review={r['id']:r['review_required'] for r in regions}
    if regions:
        xx0=max(r['box'][0] for r in regions);xx1=min(r['box'][2] for r in regions);assert xx1>xx0
        boxes={r['id']:[xx0,r['box'][1],xx1,r['box'][3]] for r in regions};width=xx1-xx0
    else:boxes={};width=0
    for d in ['originals','RGB','masks','localization']: (out/d).mkdir(exist_ok=True)
    polys={t:{} for t in periods};edge_polys={t:{} for t in periods};source_windows={};rows=[];stitched_wh=None
    for t in periods:
        crops={};geomparts={}
        for k in ids:
            H=np.asarray(byid[k]['matrices'][f'T{t}']);s=np.linalg.svd(H[:2,:2],compute_uv=False);assert np.isclose(s[0],s[1],rtol=1e-8)
            q=proj(grid(boxes[k]),H);w,h=wh[t]
            assert q[...,0].min()>=-.02 and q[...,1].min()>=-.02 and q[...,0].max()<=w-1+.02 and q[...,1].max()<=h-1+.02,(pos,t,k,q.min((0,1)),q.max((0,1)),wh[t])
            if t==1:
                x0,y0,x1,y1=boxes[k];rgb=images[t][y0:y1,x0:x1].copy()
            else:rgb=cv2.remap(images[t],q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            crops[k]=rgb;geomparts[k]=np.full(rgb.shape[:2],255,np.uint8)
            polys[t][k]=proj(corners(boxes[k]),H);edge_polys[t][k]=proj(corners(boxes[k],True),H).astype('float64')
        if regions:
            rgb=join(crops,ids,width);stitched_wh=list(rgb.shape[1::-1]);Image.fromarray(rgb).save(out/'RGB'/f'T{t}_independent_stitched_RGB.png',dpi=(600,600))
            for k in ids:Image.fromarray(crops[k]).save(out/'RGB'/f'T{t}_region_{k}_RGB.png',dpi=(600,600))
            if t==1:
                joined_mask=join(geomparts,ids,width,value=0);Image.fromarray(joined_mask).save(out/'masks'/'stitched_geometric_overlap_mask.png');Image.fromarray(joined_mask).save(out/'masks'/'effective_common_observation_mask.png')
            m=native_mask(list(polys[t].values()),wh[t]);accepted=native_mask([polys[t][k] for k in ids if not review[k]],wh[t]);domain=sbox(-.5,-.5,wh[t][0]-.5,wh[t][1]-.5)
            area=unary_union([Polygon(edge_polys[t][k]) for k in ids]).intersection(domain).area
            aa=unary_union([Polygon(edge_polys[t][k]) for k in ids if not review[k]]).intersection(domain).area if any(not review[k] for k in ids) else 0.
        else:m=np.zeros(images[t].shape[:2],np.uint8);accepted=m.copy();area=aa=None
        if entry.get('crop_source_to_regions',False) and regions:
            pp=np.concatenate(list(polys[t].values()));lo=np.floor(pp.min(0)).astype(int);hi=np.ceil(pp.max(0)).astype(int)+1;lo=np.maximum(lo,0);hi=np.minimum(hi,wh[t]);x0,y0=lo;x1,y1=hi
            Image.fromarray(images[t][y0:y1,x0:x1]).save(out/'originals'/f'T{t}_original_upper_region_RGB.png',dpi=(600,600));source_windows[f'T{t}']=[int(x0),int(y0),int(x1),int(y1)]
            m=m[y0:y1,x0:x1];accepted=accepted[y0:y1,x0:x1]
        else:
            import shutil
            shutil.copy2(paths[t],out/'originals'/paths[t].name);source_windows[f'T{t}']=[0,0,*wh[t]]
        Image.fromarray(m).save(out/'masks'/f'T{t}_native_geometric_overlap_mask.png');Image.fromarray(accepted).save(out/'masks'/f'T{t}_native_region_review_pass_mask.png')
        rows.append(dict(position=pos,period=f'T{t}',original_width=wh[t][0],original_height=wh[t][1],original_area_px2=wh[t][0]*wh[t][1],common_area_native_px2=float(area) if area is not None else None,common_area_percent=100*area/(wh[t][0]*wh[t][1]) if area is not None else None,region_review_pass_area_native_px2=float(aa) if aa is not None else None,region_review_pass_area_percent=100*aa/(wh[t][0]*wh[t][1]) if aa is not None else None,stitched_width=stitched_wh[0] if stitched_wh else 0,stitched_height=stitched_wh[1] if stitched_wh else 0,seam_area_excluded_px=24*max(len(ids)-1,0)*width,region_count=len(ids),review_required_regions=';'.join(k for k in ids if review[k]),localization_status='no_common_region' if not regions else ('review_required' if any(review.values()) else 'region_localized'),original_SHA256=sha(paths[t])))
    metadata=dict(**entry,joined_boxes_T1=boxes,stitched_RGB_wh=stitched_wh,original_wh={f'T{t}':wh[t] for t in periods},original_delivery_windows_xyxy=source_windows,area_rows=rows,polygons_native_pixel_centres={f'T{t}':{k:v.tolist() for k,v in polys[t].items()} for t in periods},area_definition='Union of transformed pixel-boundary rectangles clipped to each native original canvas, divided by native W*H. White seams and overlap duplicates excluded.',RGB_rendering='One global similarity per region; native T1 scale; original colors; no dense/nonrigid RGB warping. Missing periods are not synthesized.',manual_landmark_TRE=None)
    (out/'geometry_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2));csvwrite(out/'area_ratios.csv',rows)
    print('RGB_AREA_DONE',pos,stitched_wh,[(r['period'],round(r['common_area_percent'],2) if r['common_area_percent'] is not None else None) for r in rows],flush=True)
    return metadata

def confidence_position(work,meta,force=False):
    pos=meta['position'];out=work/'deliverable'/pos;dest=out/'confidence';dest.mkdir(exist_ok=True);dates=meta['periods'];regions=meta['regions'];boxes=meta['joined_boxes_T1'];ids=[r['id'] for r in regions]
    if not regions:return []
    if (dest/'confidence_summary.json').exists() and not force:return json.loads((dest/'confidence_summary.json').read_text())['rows']
    width=meta['stitched_RGB_wh'][0];t1w=meta['original_wh']['T1'][0];coords={r['id']:grid(boxes[r['id']]) for r in regions};cumul={k:np.ones(q.shape[:2],np.float32) for k,q in coords.items()};bottleneck={k:v.copy() for k,v in cumul.items()};supported={k:np.ones(q.shape[:2],bool) for k,q in coords.items()};rows=[];figures={}
    offsets={};yy=0
    for k in ids:offsets[k]=yy;yy+=boxes[k][3]-boxes[k][1]+24
    path_coordinates={k:(g-[boxes[k][0],boxes[k][1]]+[0,offsets[k]]).astype('float32') for k,g in coords.items()}
    for a,b in zip(dates,dates[1:]):
        ep=work/'posed_roma_evidence'/pos/f'T{a}_to_T{b}'
        assert (ep/'completed.json').exists(),('pending',pos,a,b)
        with np.load(ep/'dense_evidence.npz') as z:d={k:z[k].astype('float32') for k in z.files}
        for r in regions:
            k=r['id'];pa=path_coordinates[k];wa,wb=d['source_wh'],d['target_wh']
            cf=sample(d['source_confidence'],pa,wa)
            target=(sample(d['forward'],pa,wa)+1)*[wb[0]/2,wb[1]/2]-.5
            cb=sample(d['target_confidence'],target,wb);confidence=np.minimum(cf,cb).clip(0,1)
            back=(sample(d['backward'],target,wb)+1)*[wa[0]/2,wa[1]/2]-.5
            discrepancy=np.linalg.norm(target-pa,axis=-1);cycle=np.linalg.norm(back-pa,axis=-1)
            owner=(target[...,0]>=0)&(target[...,0]<=width-1)&(target[...,1]>=offsets[k])&(target[...,1]<=offsets[k]+boxes[k][3]-boxes[k][1]-1)
            confidence*=owner
            ok=owner&(confidence>=.5)&(discrepancy<=15)&(cycle<=10)
            cumul[k]*=confidence;bottleneck[k]=np.minimum(bottleneck[k],confidence);supported[k]&=ok
            path_coordinates[k]=target.astype('float32')
        joined=join(cumul,ids,width,value=0);bn=join(bottleneck,ids,width,value=0);valid=join({k:v.astype('uint8')*255 for k,v in supported.items()},ids,width,value=0);geom=np.array(Image.open(out/'masks'/'stitched_geometric_overlap_mask.png'))>0
        np.savez_compressed(dest/f'T1_to_T{b}_cumulative_confidence_float32.npz',confidence=joined,bottleneck_confidence=bn)
        Image.fromarray(np.rint(joined*65535).astype('uint16')).save(dest/f'T1_to_T{b}_cumulative_confidence_16bit.png')
        Image.fromarray(valid).save(out/'masks'/f'T1_to_T{b}_effective_overlap_mask.png')
        low=((joined<.5)&geom).astype('uint8')*255;Image.fromarray(low).save(out/'masks'/f'T1_to_T{b}_low_confidence_mask.png')

        full=np.zeros((meta['original_wh']['T1'][1],t1w),np.uint8);native_c=np.zeros(full.shape,np.float32)
        for k in ids:
            x0,y0,x1,y1=boxes[k];full[y0:y1,x0:x1]=supported[k].astype('uint8')*255;native_c[y0:y1,x0:x1]=cumul[k]
        if meta.get('crop_source_to_regions',False):
            x0,y0,x1,y1=meta['original_delivery_windows_xyxy']['T1'];full=full[y0:y1,x0:x1];native_c=native_c[y0:y1,x0:x1]
        Image.fromarray(full).save(out/'masks'/f'T1_to_T{b}_effective_overlap_native_T1_mask.png')
        Image.fromarray(np.rint(native_c*65535).astype('uint16')).save(dest/f'T1_to_T{b}_cumulative_confidence_native_T1_16bit.png')
        cm=plt.get_cmap('cividis')(joined)[...,:3];rgb=np.rint(cm*255).astype('uint8');rgb[~geom]=255;Image.fromarray(rgb).save(dest/f'T1_to_T{b}_cumulative_confidence_preview.png',dpi=(600,600));figures[b]=rgb
        rows.append(dict(position=pos,reference_period='T1',target_period=f'T{b}',observed_sequence='→'.join('T'+str(t) for t in dates if t<=b),geometric_common_pixels_T1=int(geom.sum()),effective_overlap_pixels_T1=int((valid>0).sum()),effective_overlap_percent_of_common=100*float((valid>0).sum())/geom.sum(),effective_overlap_percent_of_original_T1=100*float((valid>0).sum())/(t1w*meta['original_wh']['T1'][1]),low_cumulative_confidence_percent_of_common=100*float((low>0).sum())/geom.sum(),mean_cumulative_confidence_on_common=float(joined[geom].mean()),mean_bottleneck_confidence_on_common=float(bn[geom].mean())))
        print('CONFIDENCE_DONE',pos,'T'+str(b),rows[-1]['effective_overlap_percent_of_common'],flush=True)
    (dest/'confidence_summary.json').write_text(json.dumps(dict(rows=rows,input_space='Already localized, independently stitched original-color RGB at native T1 pixel scale',confidence_definition='Start at T1 query pixels. Transport coordinates through each adjacent observed-date RoMa forward field. At each edge c=min(forward confidence at transported source coordinate, reverse confidence at its predicted target). C=product c along this dense correspondence chain. Paths leaving their original observation band are zeroed. This is conditional on saved localization, not independent pose validation or calibrated probability.',effective_overlap_definition='Geometric common domain AND all transported steps remain in the same observation band, c>=0.5, step displacement<=15 T1-native pixels, and forward-backward cycle<=10 T1-native pixels. White separator rows are excluded.',low_confidence_definition='Cumulative product <0.5 inside geometric common region; white seam rows excluded.',fields='RoMa normalized dense coordinates cached as float16; coordinate transport, accumulation and saved numeric confidence float32. Dense coordinates compute QC only; RGB uses saved similarities, never dense fields.'),ensure_ascii=False,indent=2));csvwrite(dest/'confidence_summary.csv',rows)
    return rows

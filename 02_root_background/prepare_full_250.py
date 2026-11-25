"""Create native 250px patches with complete coverage of all 244+66 sources."""
import argparse, csv, json, random, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import common as c


def prepare(source, output, seed):
    source, output = Path(source).resolve(), Path(output).resolve()
    assert not (output / 'data/manifest.csv').exists(), 'Dataset already prepared'
    parents = c.csvread(source / 'data/manifest.csv')
    assert len(parents) == 310
    groups = {}
    for r in parents:
        groups.setdefault(r['group_id'], []).append(r)
    forced_train = {key for key, rs in groups.items()
                    if any(r['source'] == 'submitted_66' for r in rs)}
    keys = sorted(set(groups) - forced_train)
    rng = random.Random(seed)
    selected = None
    for _ in range(50000):
        order = keys.copy(); rng.shuffle(order)
        picked = []; count = 0
        for key in order:
            if count + len(groups[key]) <= 48:
                picked.append(key); count += len(groups[key])
        val = [r for key in picked for r in groups[key]]
        if count != 48 or any(r['source'] == 'submitted_66' for r in val):
            continue
        if set(picked) == {r['group_id'] for r in parents if r['split'] == 'val'}:
            continue
        selected = set(picked); break
    assert selected is not None
    for directory in ['data/RGB', 'data/mask', 'weights']:
        (output / directory).mkdir(parents=True, exist_ok=True)
    tiles, coverage = [], []
    for r in parents:
        r['previous_split'] = r['split']
        r['split'] = 'val' if r['group_id'] in selected else 'train'
        assert r['source'] != 'submitted_66' or r['split'] == 'train'
        rgb_path = source / 'data/RGB' / r['image_id']
        mask_path = source / 'data/mask' / r['image_id']
        assert c.sha(rgb_path) == r['RGB_sha256']
        assert c.sha(mask_path) == r['mask_sha256']
        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        h, w = mask.shape
        assert (w, h) in [(256, 256), (500, 500)]
        assert np.isin(mask, [0, 128, 255]).all()
        xs, ys = [0, w - 250], [0, h - 250]
        covered = np.zeros((h, w), np.uint8)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                name = Path(r['image_id']).stem + f'__r{j}c{i}.png'
                im, gt = rgb[y:y+250, x:x+250], mask[y:y+250, x:x+250]
                assert im.shape == (250, 250, 3) and gt.shape == (250, 250)
                covered[y:y+250, x:x+250] += 1
                ip, mp = output/'data/RGB'/name, output/'data/mask'/name
                Image.fromarray(im).save(ip); Image.fromarray(gt).save(mp)
                assert np.array_equal(np.array(Image.open(mp)), gt)
                tiles.append(dict(image_id=name, parent_image=r['image_id'], source=r['source'],
                    position=r['position'], material=r['material'], period=r['period'],
                    group_id=r['group_id'], split=r['split'], crop_x=x, crop_y=y,
                    parent_x=int(r['x']), parent_y=int(r['y']), width=250, height=250,
                    root_pixels=int((gt == 255).sum()), ignore_pixels=int((gt == 128).sum()),
                    RGB_sha256=c.sha(ip), mask_sha256=c.sha(mp)))
        assert (covered > 0).all()
        coverage.append(dict(parent_image=r['image_id'], source=r['source'], split=r['split'],
            pixels=h*w, covered_pixels=int((covered>0).sum()), root_pixels=int((mask==255).sum()),
            covered_root_pixels=int(((covered>0)&(mask==255)).sum()), tiles=4,
            minimum_exposures=int(covered.min()), maximum_exposures=int(covered.max())))
    assert len(tiles) == 1240
    assert Counter(r['split'] for r in tiles) == {'train':1048, 'val':192}
    assert sum(r['source']=='submitted_66' and r['split']=='train' for r in tiles)==264
    assert not any(r['source']=='submitted_66' and r['split']=='val' for r in tiles)
    assert not ({r['group_id'] for r in tiles if r['split']=='train'} &
                {r['group_id'] for r in tiles if r['split']=='val'})
    c.csvout(output/'data/manifest.csv', tiles)
    c.csvout(output/'data/parents.csv', parents)
    c.csvout(output/'data/coverage.csv', coverage)
    copied = {}
    for name in ['V35_original_epoch77.pt', 'V35_310_best.pt', 'V35_310_epoch100.pt']:
        src = source/'weights'/name; dst = output/'weights'/name
        shutil.copy2(src, dst); assert c.sha(src) == c.sha(dst); copied[name] = c.sha(dst)
    shutil.copy2(source/'uploaded_annotation.json', output/'uploaded_annotation.json')
    audit = dict(source_images=310, original_images=244, submitted_images=66,
        patches=1240, original_patches=976, submitted_patches=264,
        patch_shape=[250,250], resizing=False, source_pixels_discarded=0,
        annotated_root_pixels_discarded=0, all_parent_pixels_covered=True,
        train_parents=262, val_parents=48, train_patches=1048, val_patches=192,
        all_supplemental_images_in_training=True, supplemental_training_parents=66,
        supplemental_training_patches=264, supplemental_validation_images=0,
        source_splits={s:dict(Counter(r['split'] for r in tiles if r['source']==s))
                       for s in ['original_244','submitted_66']},
        train_val_group_overlap=False, seed=seed, independent_test=False,
        split_rule='Force every group containing a supplemental image into training; select48old-only parent images by intact spatial groups for internal validation; dates and child crops inherit group split',
        old_256_rule='Four native 250px corners at offsets 0 and 6; overlapping; no discarded borders',
        new_500_rule='Four native disjoint 250px quadrants at offsets 0 and 250',
        original_weights_preserved=copied, initialization='V35_310_best.pt',
        mask_codes={'0':'background','255':'all roots','128':'existing annotation ignore'},
        manifest_SHA256=c.sha(output/'data/manifest.csv'),
        parents_SHA256=c.sha(output/'data/parents.csv'),
        annotation_SHA256=c.sha(output/'uploaded_annotation.json'),
        validation_caveat='Internal validation only. Warm-start weights have historical exposure to source images. Old 256px corner patches overlap; validation must be reassembled and counted once per parent pixel.')
    c.js(output/'data_audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261125)
    a=p.parse_args();prepare(a.source,a.output,a.seed)

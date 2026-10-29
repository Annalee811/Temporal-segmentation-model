from paths import ROOT,WORK
"""Repair temporal labels on independently reconstructed root axes, not color contacts."""

import json

import pickle

import sys

from pathlib import Path

import numpy as np

from PIL import Image

from scipy import ndimage as ndi

from scipy.spatial import cKDTree

OUT = WORK/'axis'

PREV = WORK/'branch'

REFERENCE = WORK/'reference'

V83 = WORK/'joint'

PERIODS = ['T1', 'T2', 'T3', 'T4']

from engine import review_setup, reconstruct

from topology import routes

from paths import js, csvout, sha

from decode import runs

def trace(tp, paths):
    result = []
    for path in paths:
        points = []
        for segment, end in path['parts']:
            p = tp['segments'][segment]['points']
            points.extend(p if end == 0 else p[::-1])
        result.append(np.asarray(points, np.int32))
    return result

def axis_tree(points):
    xy = []; labels = []; direction = []
    for i, p in enumerate(points):
        lo = p[np.maximum(np.arange(len(p)) - 4, 0)]
        hi = p[np.minimum(np.arange(len(p)) + 4, len(p) - 1)]
        v = hi - lo; v = v / np.linalg.norm(v, axis=1).clip(1e-8)[:, None]
        xy.extend(p); labels.extend([i] * len(p)); direction.extend(v)
    return dict(tree=cKDTree(np.array(xy)), points=np.array(xy), axis=np.array(labels), direction=np.array(direction))

def correspond(tree, p, radius=4.):
    lo = p[np.maximum(np.arange(len(p)) - 4, 0)]
    hi = p[np.minimum(np.arange(len(p)) + 4, len(p) - 1)]
    v = hi - lo; v = v / np.linalg.norm(v, axis=1).clip(1e-8)[:, None]
    d, idx = tree['tree'].query(p, k=8)
    alignment = np.abs((tree['direction'][idx] * v[:, None]).sum(-1))
    valid = (d <= radius) & (alignment >= .8)
    rank = np.where(valid, alignment - d / max(radius, 1), -1e6)
    chosen = rank.argmax(1); good = valid.any(1)
    selected = idx[np.arange(len(p)), chosen]
    axis = tree['axis'][selected].copy(); axis[~good] = -1
    return good, axis, tree['points'][selected], d[np.arange(len(p)), chosen]

def body(arm, tp, limit=48):
    p = tp['segments'][arm['segment']]['points']
    q = p if arm['end'] == 0 else p[::-1]
    return q[:limit]

def main():
    OUT.mkdir(exist_ok=True); (OUT / 'full').mkdir(exist_ok=True); (OUT / 'ownership').mkdir(exist_ok=True); (OUT / 'changes').mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT / 'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3') == '0353-2-B3'
    immutable = [PREV / 'full' / (t + '_classes.png') for t in PERIODS]
    immutable += [V83 / 'joint_best.pt', REFERENCE / 'reference_provenance.json']
    hashes = {str(p): sha(p) for p in immutable}
    current = {t: np.array(Image.open(PREV / 'full' / (t + '_classes.png'))) for t in PERIODS}
    setup = review_setup()
    initial, old_junctions = reconstruct(setup, current)
    trees = {}; points_by_period = {}; states_by_period = {}; refined = {}; new_decisions = []; axisrows = []; changes = []; junctionrows = []; unresolved = []
    snapshots = {}
    for ti, t in enumerate(PERIODS):
        tp = setup['top']['topos'][t]
        selected = dict(initial[t]['hub_chosen'])


        if ti:
            for ji, hub in enumerate(tp['hubs']):
                if hub.get('disabled') or len(hub['arms']) not in [3, 4]:
                    continue
                arms = hub['arms']; options = []
                for tj in range(ti - 1, -1, -1):
                    prevt = PERIODS[tj]; prior = []
                    for a in arms:
                        q = body(a, tp); q = q[3:]
                        if len(q) < 6:
                            prior.append((-1, 0., 0.)); continue
                        good, ids, _, _ = correspond(trees[prevt], q)
                        if not good.any():
                            prior.append((-1, 0., 0.)); continue
                        values, counts = np.unique(ids[good], return_counts=True)
                        k = int(counts.argmax()); dominant = int(values[k]); purity = float(counts[k] / good.sum()); coverage = float(counts[k] / len(q))
                        prior.append((dominant, coverage, purity))
                    pairs = []
                    for a in range(len(arms)):
                        for b in range(a + 1, len(arms)):
                            ia, ca, pa = prior[a]; ib, cb, pb = prior[b]
                            if ia >= 0 and ia == ib and min(ca, cb) >= .45 and min(pa, pb) >= .8:
                                pairs.append((a, b, min(ca, cb)))
                    if len(pairs) == 1:
                        a, b, confidence = pairs[0]; others = list(set(range(len(arms))) - {a, b})
                        if len(arms) == 3 and prior[others[0]][1] <= .2:
                            options.append((confidence, [(a, b)], prevt, prior))
                        if len(arms) == 4 and all(prior[k][1] <= .2 for k in others) and -arms[others[0]]['direction'] @ arms[others[1]]['direction'] >= .5:
                            options.append((confidence, [(a, b), tuple(others)], prevt, prior))
                if options:
                    confidence, pairs, prevt, prior = max(options, key=lambda v: v[0])
                    before = selected.get(ji)
                    if before is None or {tuple(sorted(p)) for p in before[0]} != {tuple(sorted(p)) for p in pairs}:
                        selected[ji] = (pairs, 1.05 + confidence * .01)
                        new_decisions.append(dict(period=t, hub=ji, previous_period=prevt, previous_pairs=str(None if before is None else before[0]), corrected_pairs=str(pairs), arm_previous_root_evidence=str(prior), reason='two arms continue the same earlier root; separate newly visible lateral from existing mother'))
        path, rootof, accepted = routes(tp, [(j, pair, score) for j, (pair, score) in selected.items()])
        refined[t] = dict(paths=path, rootof=rootof, accepted=accepted, hub_chosen=selected)
        points = trace(tp, path); points_by_period[t] = points
        trees[t] = axis_tree(points)
        mask = current[t] > 0; core = np.zeros(mask.shape, bool)
        for h in tp['hubs']:
            if not h.get('disabled'):
                q = h['points']; core[q[:, 1], q[:, 0]] = True

        near_core = ndi.binary_dilation(core, iterations=2)
        seed = np.zeros(mask.shape, np.uint8); state_records = []; axis_at = np.full(mask.shape, -1, np.int32); path_targets = []
        for ai, p in enumerate(points):
            axis_at[p[:, 1], p[:, 0]] = ai
            labels = current[t][p[:, 1], p[:, 0]]
            interior = ~near_core[p[:, 1], p[:, 0]]
            counts = {k: int(((labels == k) & interior).sum()) for k in [1, 2, 3]}
            nodes = np.unique(setup['owners'][t][p[:,1],p[:,0]]); nodes = nodes[nodes >= 0]
            newprob = float(setup['node_new'][nodes].mean()) if len(nodes) else 0.
            record = dict(period=t, axis=ai, length=len(p), persistent=counts[1], new=counts[2], extension=counts[3], state='baseline' if ti == 0 else 'uncertain', prior_period='', prior_axis=-1, prior_support=0, prior_fraction=0., identity_affinity=0., learned_new_probability=newprob, future_aligned_support=0., x_start=int(p[0, 0]), y_start=int(p[0, 1]), x_end=int(p[-1, 0]), y_end=int(p[-1, 1]))
            if not ti or len(p) < 10:
                state_records.append(record); path_targets.append(labels.copy()); continue
            matches = []
            for tj in range(ti - 1, -1, -1):
                prevt = PERIODS[tj]; good, ids, xy, distance = correspond(trees[prevt], p)
                if good.any():
                    vals, freq = np.unique(ids[good & interior], return_counts=True)
                    for oldaxis, number in zip(vals, freq):
                        chosen = good & (ids == oldaxis)


                        longest = max([b - a for a, b, v in runs(chosen.astype(int)) if v] or [0])
                        if number >= 10 and longest >= 6:
                            q = p[chosen]
                            _, affinity = setup['track_evidence'](ti, q, [tj])
                            matches.append((float(number) * affinity, prevt, int(oldaxis), chosen, xy, distance, affinity, int(number), longest))
            state = 'uncertain'; prior_flags = np.zeros(len(p), bool); previous_axis = None
            if matches:
                _, prevt, oldaxis, flags, oldxy, distances, affinity, number, longest = max(matches, key=lambda v: v[0])
                prior_flags = flags
                record.update(prior_period=prevt, prior_axis=oldaxis, prior_support=number, prior_fraction=float(flags.mean()), identity_affinity=affinity)
                old_near_purple = int((flags & interior & (labels == 1)).sum())


                if affinity >= setup['identity_threshold'] and (old_near_purple >= 8 or flags[interior].mean() >= .6):
                    state = 'established'; previous_axis = points_by_period[prevt][oldaxis]
            temporal_support = setup['history'](ti, p)
            past_fraction = max(temporal_support[:ti] or [0.])
            future_fraction = max(temporal_support[ti+1:] or [0.])
            record['future_aligned_support'] = future_fraction


            stable_new_shaft = counts[2] >= 16 and counts[1] <= max(3, int(.08 * counts[2]))



            learned_birth = not matches and counts[2] >= 24 and newprob >= .5 and future_fraction >= .5
            if state != 'established' and (stable_new_shaft or learned_birth) and past_fraction < .2:
                state = 'new'
            record['state'] = state
            target = labels.copy(); reason = np.zeros(len(p), np.uint8)
            if state == 'new':
                take = np.isin(labels, [1, 3])
                target[take] = 2; reason[take] = 1
            elif state == 'established':
                old_positions = np.flatnonzero(prior_flags)
                first, last = int(old_positions.min()), int(old_positions.max())


                span = np.zeros(len(p), bool); span[first:last + 1] = True
                take = span & np.isin(labels, [2, 3])
                target[take] = 1; reason[take] = 2
                for a, b, boundary in [(0, first, first), (last + 1, len(p), last)]:
                    if b - a < 4: continue
                    old_tip_distance = float(min(np.linalg.norm(p[boundary] - previous_axis[0]), np.linalg.norm(p[boundary] - previous_axis[-1])))
                    novel = np.zeros(len(p), bool); novel[a:b] = True
                    extension_anchor = max([hi - lo for lo, hi, c in runs(labels[a:b]) if c == 3] or [0]) >= 6


                    if old_tip_distance <= 20 or (extension_anchor and counts[1] >= 16):
                        take = novel & (labels == 2)
                        target[take] = 3; reason[take] = 3
                    elif (novel & interior & (labels == 2)).any():




                        take = novel & (labels == 2)
                        target[take] = 3; reason[take] = 3
                        unresolved.append(dict(period=t, axis=ai, reason='old-axis distal growth assigned; exact earlier tip boundary remains uncertain', old_tip_distance=old_tip_distance, interval_pixels=b-a))
            elif counts[2] and (counts[1] or counts[3]):
                unresolved.append(dict(period=t, axis=ai, reason='root identity/state remains uncertain; kept current body labels', old_tip_distance=-1., interval_pixels=len(p)))
            changed = (target != labels) & interior
            q = p[changed]; seed[q[:, 1], q[:, 0]] = target[changed]
            for a, b, value in runs(np.where(changed,target.astype(int)*10+reason,0)):
                if not value: continue
                q = p[a:b]
                changes.append(dict(period=t, axis=ai, root_state=state, to_class=int(target[a]), skeleton_pixels=b-a, x_start=int(q[0,0]), y_start=int(q[0,1]), x_end=int(q[-1,0]), y_end=int(q[-1,1]), reason={1:'whole new-root axis: remove extension/persistent island',2:'same root already present in an earlier period: persistent span',3:'established root distal growth: remove false new label'}[int(reason[a])]))
            state_records.append(record)
            path_targets.append(target)
        states_by_period[t] = state_records



        for ji, hub in enumerate(tp['hubs']):
            if hub.get('disabled'): continue
            axes = [int(rootof[a['segment']]) for a in hub['arms']]
            pair = selected.get(ji); paired_axis = None; new_child = None
            if pair and len(hub['arms']) == 3 and len(pair[0]) == 1:
                used = set(pair[0][0]); child = next(iter(set(range(3)) - used)); paired_axis = axes[pair[0][0][0]]
                if state_records[axes[child]]['state'] == 'new' and state_records[paired_axis]['state'] == 'established': new_child = axes[child]
            q = hub['points']; yy, xx = q[:, 1], q[:, 0]
            for y, x in zip(yy, xx):
                expected = []
                for owner in sorted(set(axes)):
                    pp = points[owner]; away = ~near_core[pp[:,1],pp[:,0]]
                    use = np.flatnonzero(away)
                    if not len(use): use = np.arange(len(pp))
                    k = use[np.argmin(np.linalg.norm(pp[use]-[x,y],axis=1))]
                    value = 2 if state_records[owner]['state'] == 'new' else int(path_targets[owner][k])
                    expected.append((owner,value,float(np.linalg.norm(pp[k]-[x,y]))))
                owner = new_child
                if owner is None:


                    same = [r for r in expected if r[1] == current[t][y,x]]
                    if same: owner = min(same,key=lambda r:r[2])[0]
                    else: owner = paired_axis if paired_axis is not None else (min(expected,key=lambda r:r[2])[0] if expected else -1)
                axis_at[y,x] = owner
                if new_child is not None:
                    seed[y,x] = 2
                elif owner >= 0:
                    c = next((value for who,value,_ in expected if who==owner),int(current[t][y,x]))
                    if c in [1,2,3] and current[t][y,x] != c: seed[y,x] = c
            junctionrows.append(dict(period=t, hub=ji, arms=len(hub['arms']), incident_axes=','.join(map(str,axes)), paired_axis=-1 if paired_axis is None else paired_axis, new_lateral_axis=-1 if new_child is None else new_child))
        expected_skeleton = np.zeros(mask.shape, np.uint8)


        for ai,p in enumerate(points):
            own = (axis_at[p[:,1],p[:,0]] == ai) & ~core[p[:,1],p[:,0]]
            q = p[own]; target = path_targets[ai][own]
            expected_skeleton[q[:,1],q[:,0]] = target
            change = target != current[t][q[:,1],q[:,0]]
            q=q[change]; seed[q[:,1],q[:,0]]=target[change]
        expected_skeleton[core] = current[t][core]
        expected_skeleton[core & (seed>0)] = seed[core & (seed>0)]

        _, nearest = ndi.distance_transform_edt(axis_at < 0, return_indices=True)
        fullowner = axis_at[tuple(nearest)]; fullowner[~mask] = -1
        proposal = seed[tuple(nearest)]
        out = current[t].copy(); take = mask & (proposal > 0); out[take] = proposal[take]


        state_lookup = np.array([0 if r['state'] not in ['new','established'] else (2 if r['state']=='new' else 1) for r in state_records],np.uint8)
        owner_state = state_lookup[np.maximum(fullowner,0)]
        expected = expected_skeleton[tuple(nearest)]
        out[mask & (owner_state==2)] = 2
        use = mask & (owner_state==1) & np.isin(expected,[1,3])
        out[use] = expected[use]



        for ai,r in enumerate(state_records):
            if r['state']!='established': continue
            bad = mask & (fullowner==ai) & (out==2)
            if bad.any():
                pp=points[ai]; values=path_targets[ai]; valid=np.isin(values,[1,3])
                yy,xx=np.where(bad); tree=cKDTree(pp[valid]);_,idx=tree.query(np.c_[xx,yy]);out[yy,xx]=values[valid][idx]
        for ai,r in enumerate(state_records):
            if r['state']=='new': assert np.all(out[(fullowner==ai)&mask]==2)
            if r['state']=='established': assert not (out[(fullowner==ai)&mask]==2).any()
        assert np.array_equal(out > 0, mask)
        if not ti: out = current[t].copy()
        Image.fromarray(out).save(OUT / 'full' / (t + '_classes.png'))
        Image.fromarray((out != current[t]).astype(np.uint8)*255).save(OUT / 'changes' / (t + '_edited.png'))
        np.savez_compressed(OUT / 'ownership' / (t + '_axis_owner.npz'), axis=fullowner.astype(np.int32))
        axisrows.extend(state_records)
        snapshots[t] = dict(points=points, routes=refined[t], states=state_records)
        print('REPAIR', t, 'axes',len(points),'new',sum(r['state']=='new' for r in state_records),'established',sum(r['state']=='established' for r in state_records),'changed',int((out!=current[t]).sum()),flush=True)
    for p, digest in hashes.items(): assert sha(p) == digest
    for name, rows in [('Axis_temporal_states.csv',axisrows),('Root_interval_corrections.csv',changes),('History_corrected_junction_pairings.csv',new_decisions),('Junction_axis_ownership.csv',junctionrows),('Unresolved_root_intervals.csv',unresolved)]:
        if rows: csvout(OUT / name, rows)
    with (OUT / 'root_axis_snapshots.pkl').open('wb') as f: pickle.dump(snapshots,f)
    js(OUT / 'prediction_verification.json',dict(passed=True,no_training=True,manual_reference_classes_or_IDs_at_prediction=False,root_geometry_unchanged=True,all_four_periods_processed=True,all_prior_predictions_and_reference_unchanged=True,immutable_inputs_sha256=hashes,history_corrected_junctions=len(new_decisions),corrected_intervals=len(changes),prediction_sha256={t:sha(OUT/'full'/(t+'_classes.png')) for t in PERIODS},note='Learned ConvLSTM/GAT identity features plus explicit root-axis chronology decoding; unknown contacts are not silently converted into biological branches.'))

if __name__ == '__main__': main()

from paths import ROOT,WORK
"""Inference-only all-field branch/shaft consistency review on immutable V83."""

import hashlib

import json

import pickle

import sys

from pathlib import Path

import numpy as np

from PIL import Image

from scipy import ndimage as ndi

from scipy.spatial import cKDTree

from engine import review_setup, reconstruct

from paths import csvout, js, sha

from decode import runs

PREVIOUS = WORK/'joint/final'

OUT = WORK/'branch'

PERIODS = ['T1', 'T2', 'T3', 'T4']

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'candidate_full').mkdir(exist_ok=True)
    originals = {t: PREVIOUS / 'full' / (t + '_classes.png') for t in PERIODS}
    hashes = {str(p): sha(p) for p in originals.values()}
    hashes[str(PREVIOUS.parent / 'joint_best.pt')] = sha(PREVIOUS.parent / 'joint_best.pt')
    current = {t: np.array(Image.open(p)) for t, p in originals.items()}
    setup = review_setup()
    route, junctions = reconstruct(setup, current)
    csvout(OUT / 'All_junction_routing.csv', junctions)
    owners = setup['owners']; top = setup['top']; graph = setup['graph']
    prior_tips = {}
    for t in PERIODS:
        tp = top['topos'][t]

        sk = tp['sk']
        degree = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - sk
        yy, xx = np.where(sk & (degree == 1))
        prior_tips[t] = cKDTree(np.c_[xx, yy])
    axisrows, edits, unresolved, junction_edits = [], [], [], []
    topology_paths = {}
    for ti, t in enumerate(PERIODS):
        im = current[t]; tp = top['topos'][t]; result = im.copy()
        seed = np.zeros(im.shape, np.uint8)
        seed_reason = np.zeros(im.shape, np.uint8)
        mask = im > 0
        paths = route[t]['paths']
        points_by_axis = []
        for pi, path in enumerate(paths):
            p = []
            for s, e in path['parts']:
                q = tp['segments'][s]['points']; p.extend(q if e == 0 else q[::-1])
            p = np.asarray(p, np.int32); points_by_axis.append(p)
            if not ti or len(p) < 12:
                continue
            labels = im[p[:, 1], p[:, 0]]
            purple = int((labels == 1).sum()); green = int((labels == 2).sum()); yellow = int((labels == 3).sum())
            if not green or (not purple and not yellow):
                continue
            lo = p[np.maximum(np.arange(len(p)) - 4, 0)]
            hi = p[np.minimum(np.arange(len(p)) + 4, len(p) - 1)]
            direction = hi - lo
            direction = direction / np.linalg.norm(direction, axis=1).clip(1e-8)[:, None]
            previous_flags = []
            for tt in PERIODS[:ti]:
                dd, nn = setup['trees'][tt].query(p, k=4)
                aligned = np.abs((setup['directions'][tt][nn] * direction[:, None]).sum(-1))
                previous_flags.append(((dd <= 3.5) & (aligned >= .85)).any(1))
            observed = np.any(previous_flags, axis=0)

            observed = ndi.binary_closing(observed, structure=np.ones(3), border_value=0) | observed
            node = np.unique(owners[t][p[:, 1], p[:, 0]])
            node = node[node >= 0]
            newprob = float(setup['node_new'][node].mean()) if len(node) else 0.
            coverage, affinity = setup['track_evidence'](ti, p)
            old_enough = int(observed.sum()) >= 12
            prior_purple = int((observed & (labels == 1)).sum())


            short_old_shaft = prior_purple >= 10 and coverage >= .15 and affinity >= .65
            established = old_enough and ((coverage >= .45 and affinity >= .55 and (purple >= 12 or (observed.mean() >= .65 and newprob <= .55))) or short_old_shaft)
            newly_born = green >= 24 and purple <= 3 and observed.mean() < .1 and newprob >= .65
            axisrows.append(dict(period=t, axis=pi, length=len(p), persistent=purple, new=green, extension=yellow, earlier_visible_pixels=int(observed.sum()), prior_persistent_shaft_pixels=prior_purple, earlier_fraction=float(observed.mean()), learned_history_coverage=coverage, learned_identity_affinity=affinity, learned_new_probability=newprob, established=established, newly_born=newly_born, parts=len(path['parts'])))

            def propose(indices, value, reason):
                indices = np.asarray(indices, np.int64)
                q = p[indices]
                actual = im[q[:, 1], q[:, 0]] != value
                q = q[actual]
                if not len(q):
                    return
                seed[q[:, 1], q[:, 0]] = value
                seed_reason[q[:, 1], q[:, 0]] = 1 if value == 1 else (2 if value == 2 else 3)
                edits.append(dict(period=t, axis=pi, to_class=value, skeleton_pixels=len(q), x_start=int(q[0, 0]), y_start=int(q[0, 1]), x_end=int(q[-1, 0]), y_end=int(q[-1, 1]), reason=reason, learned_history_coverage=coverage, learned_identity_affinity=affinity))

            if established:

                supported = observed & (labels == 2)
                for a, b, v in runs(supported.astype(int)):
                    if v and b - a >= 6:
                        propose(np.arange(a, b), 1, 'established root, coherent aligned history: new-colored interval already present earlier')
                seen = np.flatnonzero(observed)


                for side, boundary in [('start', int(seen.min())), ('end', int(seen.max()))]:
                    novel = np.arange(0, boundary) if side == 'start' else np.arange(boundary + 1, len(p))
                    if len(novel) < 10:
                        continue
                    terminal_segment, terminal_end = path['parts'][0] if side == 'start' else (path['parts'][-1][0], 1 - path['parts'][-1][1])
                    end_hubs = tp['segments'][terminal_segment]['end_hubs'][terminal_end]
                    true_tip = len(end_hubs) == 0

                    endpoint = p[0] if side == 'start' else p[-1]
                    edge = min(endpoint[0], endpoint[1], im.shape[1] - 1 - endpoint[0], im.shape[0] - 1 - endpoint[1]) < 3
                    old_tip_distance = min(prior_tips[tt].query(p[boundary])[0] for tt in PERIODS[:ti])
                    body = labels[novel]
                    coherent = np.isin(body, [2, 3]).mean() >= .9 and (body == 2).sum() >= 6 and (body == 1).sum() <= 2


                    extension_anchor = any(c == 3 and b - a >= 6 for a, b, c in runs(body))
                    protected_shaft = prior_purple >= 10 and extension_anchor and affinity >= .65
                    if coherent and ((true_tip and not edge and old_tip_distance <= 8) or protected_shaft):
                        propose(novel[body == 2], 3, 'old root tip extends along the same learned root axis; new lateral is a separate unpaired axis')
                    elif coherent:
                        unresolved.append(dict(period=t, axis=pi, side=side, pixels=len(novel), reason='possible mother elongation; no sufficiently clear earlier-tip/terminal-identity evidence', old_tip_distance=float(old_tip_distance), true_terminal_tip=true_tip, endpoint_is_crop_edge=bool(edge)))
            if newly_born:

                for a, b, v in runs(labels):
                    if v == 3:
                        propose(np.arange(a, b), 2, 'new root axis with absent earlier aligned support; remove extension island without changing mother root')
                    if v == 1 and b - a <= 3 and a >= 8 and len(labels) - b >= 8 and (labels[a-8:a] == 2).mean() >= .875 and (labels[b:b+8] == 2).mean() >= .875:
                        propose(np.arange(a, b), 2, 'new root axis with two stable new anchors; short internal persistent-class island')
            if purple and green and not established and not newly_born:
                unresolved.append(dict(period=t, axis=pi, side='axis', pixels=len(p), reason='mixed old/new root identity unresolved; retained original prediction', old_tip_distance=-1., true_terminal_tip=False, endpoint_is_crop_edge=False))
        topology_paths[t] = points_by_axis

        sk = tp['sk']
        _, nearest = ndi.distance_transform_edt(~sk, return_indices=True)
        proposal = seed[tuple(nearest)]
        region = mask & (proposal > 0)
        result[region] = proposal[region]

        def arm_color(arm):
            p = tp['segments'][arm['segment']]['points']
            q = p if arm['end'] == 0 else p[::-1]

            values = result[q[:, 1], q[:, 0]]
            for offset in range(3, min(len(values) - 3, 20)):
                block = values[offset:offset + 8]
                for c in [1, 2, 3]:
                    if len(block) >= 4 and (block == c).mean() >= .875:
                        return c
            return 0

        if ti:
            for ji, hub in enumerate(tp['hubs']):
                if hub.get('disabled') or ji not in route[t]['hub_chosen']:
                    continue
                pairs, confidence = route[t]['hub_chosen'][ji]
                arms = hub['arms']; colors = [arm_color(a) for a in arms]


                used = {a for pair in pairs for a in pair}
                force_new = False
                if len(arms) == 3 and len(pairs) == 1 and len(used) == 2:
                    child = next(iter(set(range(3)) - used)); u, v = pairs[0]
                    p = arms[child]['body_points']
                    earlier = setup['history'](ti, p)

                    force_new = colors[child] == 2 and colors[u] == colors[v] and colors[u] in [1, 3] and max(earlier[:ti]) < .15
                allowed = {c for c in colors if c in [1, 2, 3]}
                if len(allowed) > 2 or not allowed:
                    continue
                q = hub['points']; yy, xx = q[:, 1], q[:, 0]
                if force_new:
                    bad = mask[yy, xx] & (result[yy, xx] != 2)
                    result[yy[bad], xx[bad]] = 2
                    if bad.any():
                        junction_edits.append(dict(period=t, hub=ji, changed_core_pixels=int(bad.sum()), reason='new daughter attachment is new; separate persistent/extension mother kept', allowed_types=','.join(map(str, sorted(allowed)))))


                elif len(colors) >= 3 and all(colors) and len(allowed) <= 2:
                    bad = ~np.isin(result[yy, xx], list(allowed)) & mask[yy, xx]
                    if bad.any():
                        available = []
                        for arm, color in zip(arms, colors):
                            if color in allowed:
                                available.append((np.linalg.norm(q[:, None, :] - arm['body_points'][None, :], axis=2).min(1), color))
                        choices = np.array([v[1] for v in available])[np.argmin(np.stack([v[0] for v in available]), axis=0)]
                        result[yy[bad], xx[bad]] = choices[bad]
                        junction_edits.append(dict(period=t, hub=ji, changed_core_pixels=int(bad.sum()), reason='exact junction core restricted to its stable incident-root classes', allowed_types=','.join(map(str, sorted(allowed)))))
        assert np.array_equal(result > 0, mask)
        if ti == 0:
            assert np.array_equal(result, im)
        Image.fromarray(result).save(OUT / 'candidate_full' / (t + '_classes.png'))
        print('REVIEW', t, 'paths', len(paths), 'changed', int((result != im).sum()), flush=True)
    for p, digest in hashes.items():
        assert sha(p) == digest
    for name, rows in [('Axis_state_review.csv', axisrows), ('Root_interval_proposals.csv', edits), ('Unresolved_root_intervals.csv', unresolved), ('Branch_and_crossing_proposals.csv', junction_edits)]:
        if rows:
            csvout(OUT / name, rows)
    with (OUT / 'root_axes.pkl').open('wb') as f:
        pickle.dump(topology_paths, f)
    js(OUT / 'candidate_verification.json', dict(passed=True, no_training=True, model_only_features_at_prediction=True, manual_IDs_or_reference_classes_prediction_input=False, all_four_full_fields=True, T1_unchanged=True, binary_geometry_unchanged=True, previous_prediction_and_checkpoint_sha256=hashes, proposed_intervals=len(edits), proposed_junctions=len(junction_edits), unresolved_intervals=len(unresolved)))

if __name__ == '__main__':
    main()

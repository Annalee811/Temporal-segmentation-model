"""Randomly initialize each frozen-root branch's temporal head; never warm start."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import common as c


def save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    c.js(temporary, value)
    os.replace(temporary, path)


def evaluate(model, data, split, device, weights, amp):
    selected = [r for r in data.pairs if r['split'] == split]
    cm = np.zeros((4, 4), np.int64)
    conditional_cm = cm.copy()
    losses = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(selected), 4):
            rr = selected[offset:offset + 4]
            items = [data.patch(r) for r in rr]
            xx = torch.from_numpy(np.stack([i[0] for i in items])).to(device).contiguous(memory_format=torch.channels_last)
            targets = np.stack([i[1] for i in items])
            gate = np.stack([data.gate[r['current']][int(r['y']):int(r['y']) + 256, int(r['x']):int(r['x']) + 256] for r in rr])
            masked = targets.copy()
            masked[~gate] = 255
            yy = torch.from_numpy(masked).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                z, boundary = model.forward_with_boundary(xx)
            value = c.loss(z.float(), yy, weights, boundary.float(), xx[:, 17])
            assert torch.isfinite(value), 'Nonfinite evaluation objective'
            losses.append((float(value), len(rr)))
            prediction = np.where(gate, z.float().argmax(1).cpu().numpy() + 1, 0).astype('uint8')
            for g, p, support in zip(targets, prediction, gate):
                cm += c.old.confusion(g, p)
                conditional = g.copy()
                conditional[~support] = 255
                conditional_cm += c.old.confusion(conditional, p)
    result = c.old.metrics(cm)
    result['loss'] = sum(v * n for v, n in losses) / sum(n for _, n in losses)
    result['macro_IoU'] = float(np.mean([r['IoU'] for r in result['classes'] if r['IoU'] is not None]))
    result['accuracy_including_background'] = float(np.trace(cm) / cm.sum())
    root = cm[1:, :]
    result['reference_root_pixel_accuracy'] = float(sum(cm[k, k] for k in [1, 2, 3]) / root.sum())
    result['conditional_on_predicted_root'] = c.old.metrics(conditional_cm)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', choices=c.MODELS, required=True)
    parser.add_argument('--threads', type=int, default=1)
    a = parser.parse_args()
    prep = json.loads((a.run / 'preparation.json').read_text())
    if (a.run / 'STOPPED_BY_USER.json').exists():
        raise SystemExit('User stop marker present; do not resume automatically')
    from dataset_identity import require_active_position
    assert require_active_position(prep['position']) == '0353-2-B3'
    out = a.run / a.model
    if (out / 'completion.json').exists():
        print('ALREADY_COMPLETE', a.model, flush=True)
        return
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(prep['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(prep['seed'])
    sampler = np.random.default_rng(prep['seed'])
    device = torch.device('cpu')
    amp = torch.cpu._is_avx512_bf16_supported()
    data = c.Data(a.run, prep['roots'], a.model)
    model = c.AxisHead().to(device, memory_format=torch.channels_last)
    model.class_multipliers = model.class_adjustment = [1., 1., 1.]
    pools = {k: [i for i,r in enumerate(data.train) if (data.targets['current'][r['current']][int(r['y']):int(r['y'])+256,int(r['x']):int(r['x'])+256] == k).any()] for k in [1,2,3]}
    assert all(pools.values()), 'A focused class has no training window'
    counts = data.counts()
    assert (counts > 0).all(), 'A training class is absent'
    weights = np.clip(np.sqrt(counts.max() / counts), 1, 12).astype('float32')
    tw = torch.from_numpy(weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=prep['learning_rate'], weight_decay=prep['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=prep['epochs'], eta_min=prep['minimum_learning_rate'])
    root = Path(prep['roots']) / a.model / 'input_manifest.json'
    config = dict(model=a.model, model_name=c.MODEL_NAMES[a.model], epochs=prep['epochs'], seed=prep['seed'], class_counts=counts.tolist(), class_weights=weights.tolist(), frozen_binary_input_SHA256=c.sha(root), reference_SHA256=c.sha(a.run / 'data/active_progress.json'), head_initialization='Random 25-channel temporal CNN, no temporal checkpoint loaded', frozen_binary_and_features=True, shared_head_architecture=True, threads=a.threads, arithmetic='BF16 convolution with FP32 objective' if amp else 'FP32', training_unique_pairs_per_epoch=len(data.train), training_draws_per_epoch=prep['training_draws_per_epoch'], focused_sampling=prep['sampling'], hard_temporal_decoding=False, batch_size=4, objective='Weighted CE + 0.5 class Dice + 0.2 boundary BCE + 0.05 local consistency + 0.3 skeleton-path loss', reported_loss='Training: augmented update batches, pair-count-weighted epoch mean; validation: unaugmented fixed views, same class weights/objective, predicted-root support; end-to-end metrics include missed roots', metric_reference='Released training annotation set A; internal spatial agreement', initialization_scope='Temporal head only; binary predictions frozen; no learned routing; features newly computed from RGB/binary only', optimizer='AdamW', initial_learning_rate=prep['learning_rate'], min_learning_rate=prep['minimum_learning_rate'], weight_decay=prep['weight_decay'], scheduler='100-epoch cosine annealing, step after each epoch', selection='Maximum validation end-to-end macro Dice; initial epoch 0 excluded', no_test_selection=True)
    config['metric_reference'] = 'Released training annotation set A; internal spatial agreement, not independent tube accuracy'
    config['annotation_semantics'] = prep['annotation_semantics']
    config.update(GAT=False, old_temporal_checkpoint_loaded=False, old_temporal_probability_used=False, learned_router_used=False, input_channels=25, outer_partition=prep['outer_partition'], evaluation_scope=prep['evaluation_scope'])
    if (out / 'config.json').exists():
        assert json.loads((out / 'config.json').read_text()) == config, 'Resume config mismatch'
    else:
        write_json(out / 'config.json', config)
    def checkpoint(epoch):
        return dict(state_dict=model.state_dict(), epoch=epoch, config=config, class_multipliers=[1., 1., 1.], class_adjustment=[1., 1., 1.])
    history, draws, first, best, best_epoch, elapsed_before = [], [], 1, -1., 0, 0.
    tic_run = time.monotonic()
    if (out / 'resume.pt').exists():
        ck = torch.load(out / 'resume.pt', map_location='cpu', weights_only=True)
        assert ck['config'] == config
        model.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        schedule.load_state_dict(ck['scheduler'])
        rng.bit_generator.state = ck['augmentation_rng']
        sampler.bit_generator.state = ck['sampler_rng']
        torch.set_rng_state(ck['torch_rng'])
        history, draws, first = ck['history'], ck['draws'], ck['epoch'] + 1
        best, best_epoch, elapsed_before = ck['best'], ck['best_epoch'], ck['elapsed_seconds']
        print('RESUMED', a.model, ck['epoch'], flush=True)
    else:
        save(checkpoint(0), out / 'initial_head.pt')
        baseline = {s: evaluate(model, data, s, device, tw, amp) for s in ['train', 'val']}
        write_json(out / 'epoch0_metrics.json', baseline)
        history.append(dict(epoch=0, training_loss=baseline['train']['loss'], validation_loss=baseline['val']['loss'], learning_rate=optimizer.param_groups[0]['lr'], validation_macro_Dice=baseline['val']['macro_Dice'], validation_macro_IoU=baseline['val']['macro_IoU'], validation_accuracy=baseline['val']['accuracy_including_background'], validation_root_accuracy=baseline['val']['reference_root_pixel_accuracy'], seconds=time.monotonic() - tic_run, phase='untrained_fixed_evaluation'))
        for metric in baseline['val']['classes']:
            for field in ['Precision','Recall','Dice','IoU']:
                history[-1]['validation_'+str(metric['class_id'])+'_'+field] = metric[field]
        c.csvout(out / 'history.csv', history)
        save(dict(**checkpoint(0), optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(), augmentation_rng=rng.bit_generator.state, sampler_rng=sampler.bit_generator.state, torch_rng=torch.get_rng_state(), history=history, draws=draws, best=best, best_epoch=best_epoch, elapsed_seconds=time.monotonic() - tic_run), out / 'resume.pt')
        print('INITIALIZED', a.model, baseline['val']['macro_Dice'], flush=True)
    for epoch in range(first, prep['epochs'] + 1):
        tic = time.monotonic()
        model.train()
        order = sampler.permutation(len(data.train)).tolist()
        focused_classes = [1,2,3,3]
        order += [int(sampler.choice(pools[focused_classes[i % 4]])) for i in range(prep['focused_draws_per_epoch'])]
        sampler.shuffle(order)
        assert len(order) == prep['training_draws_per_epoch']
        losses = []
        lr_used = optimizer.param_groups[0]['lr']
        for offset in range(0, len(order), 4):
            if (a.run/'STOPPED_BY_USER.json').exists():
                raise SystemExit('Stopped by user; last complete epoch saved; incomplete epoch will replay')
            rr = [data.train[i] for i in order[offset:offset + 4]]
            items = [data.patch(r, True, rng) for r in rr]
            xx = torch.from_numpy(np.stack([i[0] for i in items])).to(device).contiguous(memory_format=torch.channels_last)
            yy = torch.from_numpy(np.stack([i[1] for i in items])).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                z, boundary = model.forward_with_boundary(xx)
            value = c.loss(z.float(), yy, tw, boundary.float(), xx[:, 17])
            assert torch.isfinite(value), 'Nonfinite training objective'
            value.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            losses.append((float(value.detach()), len(rr)))
            for j, (row, item) in enumerate(zip(rr, items)):
                draws.append(dict(epoch=epoch, draw=offset + j, pair_id=row['pair_id'], **item[2]))
            if offset % 40 == 0:
                write_json(out / 'status.json', dict(stage='training', epoch_completed=epoch-1, epoch_in_progress=epoch, completed_pairs=offset+len(rr), pairs_per_epoch=len(order), pid=os.getpid(), updated_unix=time.time()))
        val = evaluate(model, data, 'val', device, tw, amp)
        if val['macro_Dice'] > best:
            best, best_epoch = val['macro_Dice'], epoch
            save(checkpoint(epoch), out / 'best_temporal_head.pt')
            write_json(out / 'selected_validation.json', val)
        schedule.step()
        rec = dict(epoch=epoch, training_loss=sum(v*n for v,n in losses)/sum(n for _,n in losses), validation_loss=val['loss'], learning_rate=lr_used, validation_macro_Dice=val['macro_Dice'], validation_macro_IoU=val['macro_IoU'], validation_accuracy=val['accuracy_including_background'], validation_root_accuracy=val['reference_root_pixel_accuracy'], seconds=time.monotonic()-tic, phase='training')
        for metric in val['classes']:
            for field in ['Precision','Recall','Dice','IoU']:
                rec['validation_'+str(metric['class_id'])+'_'+field] = metric[field]
        history.append(rec)
        elapsed = elapsed_before + time.monotonic() - tic_run
        save(dict(**checkpoint(epoch), optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(), augmentation_rng=rng.bit_generator.state, sampler_rng=sampler.bit_generator.state, torch_rng=torch.get_rng_state(), history=history, draws=draws, best=best, best_epoch=best_epoch, elapsed_seconds=elapsed), out / 'resume.pt')
        c.csvout(out / 'history.csv', history)
        c.csvout(out / 'training_draws.csv', draws)
        write_json(out / 'status.json', dict(stage='training', epoch_completed=epoch, epochs_total=prep['epochs'], best_epoch=best_epoch, best_validation_macro_Dice=best, last_epoch_seconds=rec['seconds'], elapsed_seconds=elapsed, pid=os.getpid(), updated_unix=time.time()))
        print('EPOCH', a.model, epoch, 'train', round(rec['training_loss'],6), 'val_loss',round(val['loss'],6),'val_Dice',round(val['macro_Dice'],6),'seconds',round(rec['seconds'],2),flush=True)
    selected = torch.load(out / 'best_temporal_head.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(selected['state_dict'])
    test = evaluate(model, data, 'test', device, tw, amp)
    write_json(out / 'test_metrics.json', test)
    assert c.sha(root) == config['frozen_binary_input_SHA256']
    for item in prep['immutable_inputs']:
        assert c.sha(Path(item['path'])) == item['sha256'], 'Frozen mask/reference was modified'
    write_json(out / 'completion.json', dict(state='complete', epochs_completed=prep['epochs'], selected_epoch=best_epoch, selected_validation_macro_Dice=best, test_macro_Dice=test['macro_Dice'], checkpoint_sha256=c.sha(out/'best_temporal_head.pt'), root_weights_unchanged=True, seconds=elapsed_before+time.monotonic()-tic_run))
    write_json(out / 'status.json', dict(stage='complete',epoch_completed=prep['epochs'],selected_epoch=best_epoch,updated_unix=time.time()))
    print('COMPLETE',a.model,best_epoch,test['macro_Dice'],flush=True)


if __name__ == '__main__':
    main()

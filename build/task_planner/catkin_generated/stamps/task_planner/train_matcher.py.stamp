#!/usr/bin/env python3
"""Train an HGT Ship--Task ranking model from ALNS teacher data."""
import argparse, json, random
from pathlib import Path
import numpy as np
from pyg_adapter import require_pyg
from hgt_matcher import build_model


def record_to_data(record):
    torch, HeteroData = require_pyg(); data = HeteroData()
    data['ship'].x = torch.tensor(record['ship_x'], dtype=torch.float)
    data['task'].x = torch.tensor(record['task_x'], dtype=torch.float)
    data['road'].x = torch.tensor(record['road_x'], dtype=torch.float)
    for src, rel, dst, e, f in [('road','connects','road','rr_edges','rr_feat'), ('ship','at','road','sr_edges','sr_feat'),
                                ('task','uses','road','tr_edges','tr_feat'), ('task','related','task','tt_edges','tt_feat')]:
        edge_index = torch.tensor(record[e], dtype=torch.long); edge_attr = torch.tensor(record[f], dtype=torch.float)
        data[(src,rel,dst)].edge_index = edge_index; data[(src,rel,dst)].edge_attr = edge_attr
        if src != dst:
            data[(dst,'rev_'+rel,src)].edge_index = edge_index.flip(0); data[(dst,'rev_'+rel,src)].edge_attr = edge_attr
    # Candidate pairs are used by the scorer and also kept as a graph relation.
    pairs = torch.tensor(record['pairs'], dtype=torch.long)
    data[('ship','can_serve','task')].edge_index = pairs.t().contiguous()
    data[('ship','can_serve','task')].edge_attr = torch.tensor(record['pair_features'], dtype=torch.float)
    data[('task','rev_can_serve','ship')].edge_index = pairs.t().flip(0).contiguous()
    data[('task','rev_can_serve','ship')].edge_attr = torch.tensor(record['pair_features'], dtype=torch.float)
    targets = record.get('targets', record['labels'])
    return data, pairs, torch.tensor(record['pair_features'], dtype=torch.float), torch.tensor(record['labels'], dtype=torch.float), torch.tensor(targets, dtype=torch.float)


def recall_at_k(logits, pairs, labels, k=3):
    values = logits.detach().cpu().numpy(); pairs = pairs.cpu().numpy(); labels = labels.cpu().numpy()
    hits = total = 0
    for tid in np.unique(pairs[:, 1]):
        idx = np.where(pairs[:, 1] == tid)[0]
        if labels[idx].sum() <= 0: continue
        top = idx[np.argsort(values[idx])[-k:]]
        hits += int(labels[top].sum() > 0); total += 1
    return hits / max(total, 1)


def marginal_ranking_loss(logits, pairs, targets, torch):
    """Pairwise listwise surrogate within each task's candidate ships."""
    losses = []
    for tid in torch.unique(pairs[:, 1]):
        idx = torch.where(pairs[:, 1] == tid)[0]
        if len(idx) < 2:
            continue
        score_delta = logits[idx].unsqueeze(1) - logits[idx].unsqueeze(0)
        target_delta = targets[idx].unsqueeze(1) - targets[idx].unsqueeze(0)
        mask = target_delta > 1e-5
        if mask.any():
            losses.append(torch.nn.functional.softplus(-score_delta[mask]).mean())
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True, nargs='+'); p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=40); p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--k', type=int, default=3)
    p.add_argument('--seed', type=int, default=7); args = p.parse_args()
    torch, _ = require_pyg(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records = []
    for path in args.data:
        records.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    if len(records) < 2: raise ValueError('need at least two complete scenario records')
    random.shuffle(records); cut = max(1, int(len(records) * .8)); train, valid = records[:cut], records[cut:]
    first, _, _, _, _ = record_to_data(train[0]); model = build_model(first.metadata())
    # Materialize lazy input layers before optimizer creation.
    d, pair, feat, _, _ = record_to_data(train[0]); model(d, pair, feat)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for rec in train:
            data, pairs, features, labels, targets = record_to_data(rec); logits = model(data, pairs, features)
            pos = (len(labels) - labels.sum()) / labels.sum().clamp(min=1)
            owner_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos)
            value_loss = torch.nn.functional.smooth_l1_loss(torch.sigmoid(logits), targets)
            rank_loss = marginal_ranking_loss(logits, pairs, targets, torch)
            # Owner labels remain a weak teacher signal; the primary loss now
            # ranks counterfactual marginal schedule value within each task.
            loss = 0.25 * owner_loss + 0.50 * value_loss + rank_loss
            optim.zero_grad(); loss.backward(); optim.step(); losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            model.eval(); scores=[]
            with torch.no_grad():
                for rec in valid:
                    data,pairs,features,labels,_=record_to_data(rec); scores.append(recall_at_k(model(data,pairs,features),pairs,labels,args.k))
            print(f'epoch={epoch} loss={np.mean(losses):.4f} valid_owner_recall@{args.k}={np.mean(scores):.3f}', flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'metadata': first.metadata(), 'k': args.k,
                'target_definition': train[0].get('target_definition', 'legacy')}, output)
    print(f'Wrote: {output}')


if __name__ == '__main__': main()

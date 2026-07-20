#!/usr/bin/env python3
"""Optional learned scorer with a safe deterministic fallback."""
from pathlib import Path


class LearnedCandidateScorer:
    def __init__(self, model_path=None, k=5):
        self.model_path = Path(model_path) if model_path else None
        self.k = k
        self.enabled = False
        self.model = None
        self.last_confidence = {}
        self.last_ranked_scores = {}
        if self.model_path and self.model_path.exists():
            try:
                import torch
                from hgt_matcher import build_model
                checkpoint = torch.load(self.model_path, map_location='cpu')
                self.model = build_model(checkpoint['metadata'])
                # Materialise PyG lazy input layers with a first real graph in rank().
                self.checkpoint = checkpoint
                self.enabled = True
            except Exception:
                self.enabled = False

    def rank(self, graph):
        """Return task_id -> ranked ship IDs; deterministic edge-score fallback."""
        if self.enabled:
            try:
                import torch
                from pyg_adapter import to_heterodata
                data = to_heterodata(graph)
                pairs = torch.tensor(graph.st_edges.T, dtype=torch.long)
                features = torch.tensor(graph.st_feat, dtype=torch.float)
                self.model(data, pairs, features)  # initialise lazy layers
                self.model.load_state_dict(self.checkpoint['state_dict'])
                self.model.eval()
                with torch.no_grad(): scores = self.model(data, pairs, features).cpu().numpy()
                return self._rank_scores(graph, scores)
            except Exception:
                self.enabled = False
        return self._rank_scores(graph, graph.st_feat[:, 6])

    def _rank_scores(self, graph, scores):
        ranked = {tid: [] for tid in graph.task_ids}
        for tid in graph.task_ids:
            tj = graph.task_idx[tid]; candidates = []
            for k in range(graph.st_edges.shape[1]):
                si, j = graph.st_edges[:, k]
                if int(j) == tj:
                    candidates.append((float(scores[k]), graph.ship_ids[int(si)]))
            ordered = sorted(candidates, reverse=True)
            self.last_ranked_scores[tid] = [(sid, float(score)) for score, sid in ordered]
            ranked[tid] = [sid for _, sid in ordered[:self.k]]
            self.last_confidence[tid] = float(ordered[0][0] - ordered[1][0]) if len(ordered) > 1 else 0.0
        return ranked

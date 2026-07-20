# Static GNN matching

1. Generate teacher data with `dataset_builder.py`; the teacher is the current
   high-budget Graph+ALNS solver.
2. Train an HGT Ship--Task ranking model.
3. Use its Top-K result only as an ALNS candidate filter; hard constraints and
   a rules-based fallback remain mandatory.

Install dependencies in a dedicated virtual environment using
`requirements-learning.txt`.  Split train/validation/test by complete scenario
seed, never by snapshots from the same scenario.

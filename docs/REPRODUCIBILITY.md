# Reproducibility

The recommended reproduced configuration is:

- trajectory model: `baller2vec`
- event predictor: `disentangled`
- dynamic ratio: `--dynamic_weight 1.0`
- context: enabled
- Voronoi spatial prior: enabled when Shapely is available

Train directly with:

```bash
python main.py \
  --pt_dir /path/to/dataloader_140games_validbatch \
  --sample_ml_dir /path/to/data_140games_weighted/ml_3894534_1020746 \
  --output_dir outputs/baller2vec_ratio1 \
  --games 140 --epochs 80 --batch_size 4 --num_workers 0 \
  --traj_model baller2vec --event_head disentangled \
  --dynamic_weight 1.0 --use_context --use_voronoi_spatial
```

Evaluate the released checkpoint with:

```bash
python eval_checkpoint.py \
  --pt_dir /path/to/dataloader_140games_validbatch \
  --sample_ml_dir /path/to/data_140games_weighted/ml_3894534_1020746
```

`eval_checkpoint.py` uses the bundled Baller2Vec–DisTraj checkpoint and writes metrics under `outputs/eval_checkpoint/metrics/` by default. Use `--checkpoint` or `--output_dir` to override either path.

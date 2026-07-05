# Reproducibility

The recommended reproduced configuration is:

- trajectory model: `baller2vec`
- event predictor: `disentangled`
- dynamic ratio: `--dynamic_weight 1.0`
- context: enabled
- Voronoi spatial prior: enabled when Shapely is available

Use:

```bash
PT_DIR=/path/to/dataloader_140games_validbatch SAMPLE_ML_DIR=/path/to/data_140games_weighted/ml_3894534_1020746 scripts/train_baller2vec_ratio1.sh
```

For an existing checkpoint:

```bash
PT_DIR=/path/to/dataloader_140games_validbatch SAMPLE_ML_DIR=/path/to/data_140games_weighted/ml_3894534_1020746 CHECKPOINT=/path/to/checkpoint.pt scripts/eval_checkpoint.sh
```

All generated checkpoints and metrics are written under `OUTPUT_DIR`, which defaults to `outputs/...`.

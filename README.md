# DisTraj-Event

DisTraj-Event is a PyTorch implementation for joint soccer trajectory and event prediction. The code predicts multi-agent future trajectories, predicts the target player and event time, and evaluates the event position by reading the predicted target-player trajectory at the predicted event time.

The released training objective optimizes trajectory prediction, target-player prediction, event-time prediction, and the event-guided disentanglement losses. Event-position metrics are trajectory-anchored. Event-type logits are kept in the model interface for compatibility with earlier checkpoints, but event-type loss is not optimized by default.

## Repository Layout

- `main.py`: training and checkpoint evaluation entrypoint.
- `model_traj/`: Baller2Vec, STAR, LSTM, VRNN, and constant-velocity trajectory modules.
- `model_event/`: linear, transformer, sequence, TGN, Volterra, and disentangled event predictors.
- `train/`: curriculum training, metrics, interpolation, and distributed helpers.
- `utils/`: preprocessing utilities for building `.pt` dataloader shards.
- `visualize/`: plotting utilities for trajectories, event probabilities, and representation diagnostics.
- `scripts/`: reproducible command templates for local training, checkpoint evaluation, and Slurm.
- `docs/`: data format and reproducibility notes.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default wheel is not appropriate for your machine.

## Data Inputs

Training and evaluation require two paths:

- `--pt_dir`: dataloader root with `train/`, `val/`, and `test/` subdirectories containing `.pt` batch files.
- `--sample_ml_dir`: one `ml_*` feature directory containing `edges_raw_2d.npz` and `nodes_dynamic_2d.npz`, used to recover feature column names.

For checkpoint evaluation, only the `test/` split under `--pt_dir` is required. See `docs/DATA_FORMAT.md` for expected fields.

## Local Training

Recommended reproduction configuration for the best Baller2Vec ratio-1.0 setting:

```bash
PT_DIR=/path/to/dataloader_140games_validbatch SAMPLE_ML_DIR=/path/to/data_140games_weighted/ml_3894534_1020746 OUTPUT_DIR=outputs/baller2vec_ratio1 scripts/train_baller2vec_ratio1.sh
```

Equivalent direct command:

```bash
python main.py   --pt_dir /path/to/dataloader_140games_validbatch   --sample_ml_dir /path/to/data_140games_weighted/ml_3894534_1020746   --output_dir outputs/baller2vec_ratio1   --games 140   --epochs 80   --batch_size 4   --num_workers 0   --traj_model baller2vec   --event_head disentangled   --dynamic_weight 1.0   --use_context   --use_voronoi_spatial
```

## Checkpoint Evaluation

```bash
PT_DIR=/path/to/dataloader_140games_validbatch SAMPLE_ML_DIR=/path/to/data_140games_weighted/ml_3894534_1020746 CHECKPOINT=/path/to/best_baller2vec_ratio1.pt OUTPUT_DIR=outputs/eval_baller2vec_ratio1 scripts/eval_checkpoint.sh
```

This loads the checkpoint, evaluates the test split, prints trajectory and event metrics, and writes JSON metrics under `OUTPUT_DIR/metrics/`.

## Slurm Reproduction

```bash
PT_DIR=/path/to/dataloader_140games_validbatch SAMPLE_ML_DIR=/path/to/data_140games_weighted/ml_3894534_1020746 PYTHON=/path/to/python NPROC_PER_NODE=4 sbatch scripts/slurm_baller2vec_ratio1.sh
```

To evaluate an existing checkpoint through Slurm, also pass `CHECKPOINT=/path/to/checkpoint.pt`.

## Build Dataloader Shards

If you already have sequence files and per-match graph features:

```bash
python -m utils.gin6_shard   --seqs_dir /path/to/sequences   --ml_root /path/to/ml_root   --output_dir /path/to/dataloader_root   --batch_size 64
```

## Notes

- Large data, checkpoints, logs, TensorBoard runs, and generated figures are intentionally ignored by `.gitignore`.
- The code does not include proprietary match data. Users must prepare data according to their own data license.
- `--n_event_types` defaults to `5` for compatibility with existing checkpoints. Change it only when training from scratch with a matching event-head shape.

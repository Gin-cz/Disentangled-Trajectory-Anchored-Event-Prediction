# DisTraj-Event

DisTraj-Event is a PyTorch implementation for joint soccer trajectory and event prediction. The code predicts multi-agent future trajectories, predicts the target player and event time, and evaluates the event position by reading the predicted target-player trajectory at the predicted event time.

The released training objective optimizes trajectory prediction, target-player prediction, event-time prediction, and the event-guided disentanglement losses. Event-position metrics are trajectory-anchored. Event-type logits are kept in the model interface for compatibility with earlier checkpoints, but event-type loss is not optimized by default.

## Repository Layout

- `main.py`: training and checkpoint evaluation entrypoint.
- `eval_checkpoint.py`: evaluation entrypoint for the released checkpoint.
- `checkpoints/`: released Baller2Vec–DisTraj model weights.
- `model_traj/`: Baller2Vec, STAR, LSTM, VRNN, and constant-velocity trajectory modules.
- `model_event/`: main and baseline event predictors; nonlinear, HSIC/GRL, patch-wise, and gated ablations are consolidated in `disentangled_nonlinear.py`.
- `train/`: curriculum training, metrics, interpolation, and distributed helpers.
- `utils/`: event–tracking preprocessing, graph/sequence construction, and sharded dataset loading.
- `visualize/`: trajectory, event-probability, and representation visualizations.
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

For checkpoint evaluation, only the `test/` split under `--pt_dir` is required. See `docs/DATA_FORMAT.md` for expected fields.


## Build Dataloader Shards

If you already have sequence files and per-match graph features:

```bash
python -m utils.dataset \
  --seqs_dir /path/to/sequences \
  --ml_root /path/to/ml_root \
  --output_dir /path/to/dataloader_root \
  --batch_size 64
```

## Notes

- Large data, locally trained checkpoints, logs, TensorBoard runs, and generated figures are intentionally ignored by `.gitignore`; the documented released checkpoint is included.
- The code does not include proprietary match data. Users must prepare data according to their own data license.
- `--n_event_types` defaults to `5` for compatibility with existing checkpoints. Change it only when training from scratch with a matching event-head shape.

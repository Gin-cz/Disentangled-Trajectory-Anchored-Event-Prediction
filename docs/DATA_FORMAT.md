# Data Format

`main.py` consumes a dataloader root passed through `--pt_dir`:

```text
dataloader_root/
  train/**/*.pt
  val/**/*.pt
  test/**/*.pt
```

Each `.pt` file is a pre-collated batch dictionary. The training loop expects at least these keys:

- `ctx_dyns`: context dynamic node features.
- `ctx_edges`: context event or edge features.
- `ctx_seqX`: context trajectories with player and ball positions.
- `fut_seqX`: future trajectory tensor.
- `fut_players`: future target-player index labels.
- `fut_positions`: future event-position labels.
- `ctx_labels_22`: player labels for context frames.
- `fut_labels_22`: player labels for future frames.

`--sample_ml_dir` must contain:

- `edges_raw_2d.npz` with a `columns` array.
- `nodes_dynamic_2d.npz` with a `columns` array.

These files are used to reconstruct feature layouts. They can come from any match as long as the feature schema matches the `.pt` batches.

## Building `.pt` Batches

Use `utils.dataset` after preparing sequence `.npz` files and per-match `ml_*` feature folders:

```bash
python -m utils.dataset \
  --seqs_dir /path/to/sequences \
  --ml_root /path/to/ml_root \
  --output_dir /path/to/dataloader_root
```

The preprocessing modules use descriptive names: `preprocessing.py` aligns event and tracking data and constructs synchronized node/edge features, `graph_features.py` builds graph tensors, `sequence_builder.py` creates fixed-length trajectory sequences, and `player_features.py` derives static player metadata. Their paths are controlled by environment variables such as `DISTRAJ_RAW_DATA_ROOT`, `DISTRAJ_ID_MAPPING`, `DISTRAJ_ALIGNED_DIR`, `DISTRAJ_NODE_EDGE_DIR`, `DISTRAJ_ML_ROOT`, and `DISTRAJ_SEQUENCE_DIR`.

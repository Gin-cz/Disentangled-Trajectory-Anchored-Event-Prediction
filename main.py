# -*- coding: utf-8 -*-
"""
Main Training Script with Curriculum Learning + Multi-Model Support

Supports multiple trajectory prediction models:
- Baller2Vec (with/without context)
- STAR5 (with/without context)
- LSTM (with/without context)

Training uses curriculum learning by default.
"""
from __future__ import annotations
import os
os.environ["TMPDIR"] = "/tmp"
os.environ["TEMP"] = "/tmp"
os.environ["TMP"] = "/tmp"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from train.curriculum_schedule import CurriculumScheduler
from torch.utils.tensorboard import SummaryWriter

import argparse
from glob import glob
from tqdm.auto import tqdm
import random
import numpy as np
import csv
import json
import time
import warnings
import datetime

import torch
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy("file_descriptor")
except RuntimeError:
    pass
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.gin6_shard import MatchDataset, make_adaptive_collate
from net.context_encoder import ContextEncoder
from net.voronoi_prior import TgtPlayerScorer

from model_traj.ball_predictor import BallStepHead
from model_event.event_predictor import ParallelEventPredictor_Unified

from train.train import step_batch_curriculum, evaluate_curriculum
from train.ddp import is_dist_avail_and_initialized, get_rank, get_world_size
from train.early_stopping import EarlyStopping

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

# =================== Default Configuration ===================
DEFAULTS = dict(
    batch_size=64,
    n_agents=22,
    n_home=None,
    n_away=None,
    ball_idx=None,
    field_L=105.0,
    field_W=68.0,
    max_side_id=23,
    x_dim=2,
    h_dim=128,
    z_dim=16,
    n_layers=1,
    burn_in=20,
    fut_T=64,
    short_fut_T=28,         #   2.8s @ 10Hz
    w_short_traj=0.5,       #  Short-term trajectory weight
    lr=3e-4,
    beta_kl=0.2,
    w_tgt=10,
    w_eidx=2,
    w_epos=0,
    w_traj=1.0,
    epochs=30,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    C=10,
    H=1,
    LMU_D=64,
    F_LMU_IN=2,
    F_LMU=64,
    LMU_H=128,
    volterra_Q=2,
    volterra_scales=2,
    traj_model='star',
    use_context=True,
    predict_ball=False,
    w_ball=1.0,
    use_event_emb=False,
    # ✅ Curriculum defaults (更新为新版默认值)
    curriculum_type='cosine',
    traj_short_only_epochs=20,  # 阶段1: 短轨迹
    traj_warmup_epochs=10,      # 阶段2: 变长
    event_start_epoch=50,       # 阶段3: 事件开始
    event_warmup_epochs=10,     # 阶段4: 事件变大
    
    initial_event_weight=0.0,
    final_event_weight=1.0,
    initial_disentangle_weight=0.0,
    final_disentangle_weight=0.1,
)

def parse_args():
    p = argparse.ArgumentParser(description="Train with Curriculum Learning")
    p.add_argument('--games', type=int, default=10)
    p.add_argument('--pt_dir', type=str, default=None)
    p.add_argument('--sample_ml_dir', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='outputs',
                   help='Directory for checkpoints, TensorBoard logs, and metric files.')
    p.add_argument('--run_name', type=str, default=None,
                   help='Optional run name prefix. Defaults to model/config name.')
    p.add_argument('--resume', '--checkpoint', dest='resume', type=str, default=None,
                   help='Checkpoint path to load before training or evaluation.')
    p.add_argument('--eval_only', action='store_true',
                   help='Load --resume and evaluate the test split without training.')
    p.add_argument('--strict_resume', action='store_true',
                   help='Require an exact state-dict match when loading --resume.')
    p.add_argument('--distributed', action='store_true',
                   help='Force distributed initialization. torchrun also enables it automatically.')
    p.add_argument('--n_event_types', type=int, default=5,
                   help='Number of event-type logits kept for checkpoint compatibility. Event-type loss is not optimized by default.')
    p.add_argument('--limit_train_files', type=int, default=0,
                   help='Optional smoke-test limit for train .pt files. 0 uses all files.')
    p.add_argument('--limit_val_files', type=int, default=0,
                   help='Optional smoke-test limit for validation .pt files. 0 uses all files.')
    p.add_argument('--limit_test_files', type=int, default=0,
                   help='Optional smoke-test limit for test .pt files. 0 uses all files.')
    p.add_argument('--predict_ball', action='store_true', default=DEFAULTS['predict_ball'])
    p.add_argument('--w_ball', type=float, default=DEFAULTS['w_ball'])
    p.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    p.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    p.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    p.add_argument('--device', type=str, default=DEFAULTS['device'])
    p.add_argument('--C', type=int, default=DEFAULTS['C'])
    p.add_argument('--H', type=int, default=DEFAULTS['H'])
    p.add_argument('--n_agents', type=int, default=DEFAULTS['n_agents'])
    p.add_argument('--n_home', type=int, default=DEFAULTS['n_home'])
    p.add_argument('--n_away', type=int, default=DEFAULTS['n_away'])
    p.add_argument('--ball_idx', type=int, default=DEFAULTS['ball_idx'])
    p.add_argument('--field_L', type=float, default=DEFAULTS['field_L'])
    p.add_argument('--field_W', type=float, default=DEFAULTS['field_W'])
    p.add_argument('--max_side_id', type=int, default=DEFAULTS['max_side_id'])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--pin_memory', action='store_true', default=False)
    p.add_argument('--disable_dyn_ctx', action='store_true', default=False)
    p.add_argument('--disable_edge_ctx', action='store_true', default=False)
    p.add_argument('--disable_prior', action='store_true', default=False)
    p.add_argument('--test_every_n_epochs', type=int, default=20)

    p.add_argument('--short_fut_T', type=int, default=DEFAULTS['short_fut_T'],
                   help='Short-term prediction horizon (frames)')
    p.add_argument('--w_short_traj', type=float, default=DEFAULTS['w_short_traj'],
                   help='Weight for short-term trajectory loss')
    p.add_argument('--initial_short_weight', type=float, default=1.0,
                   help='Initial short-term weight (Stage 1: short-only)')
    p.add_argument('--final_short_weight', type=float, default=0.5,
                   help='Final short-term weight (Stage 3: balanced)')
    p.add_argument('--dynamic_weight', type=float, default=0.5,
               help='Weight for dynamic features in disentangled head '
                    '(0.0=intent only, 0.5=balanced, 1.0=dynamic only)')
    
    p.add_argument('--use_hsic', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help='Ablation: Whether to use HSIC in patch disentangled head')
    p.add_argument('--decay_rate', type=float, default=0.5,
                   help='Temporal decay rate for HSIC in patch version')
    # Volterra
    p.add_argument('--volterra_Q', type=int, default=DEFAULTS['volterra_Q'])
    p.add_argument('--volterra_scales', type=int, default=DEFAULTS['volterra_scales'])
    p.add_argument('--lightweight', action='store_true', default=False)
    
    # Model selection
    p.add_argument('--event_head', type=str, default='transformer',
                   choices=['transformer', 'linear', 'seq2event', 'nmstpp', 'disentangled', 'disentangled_gated',
                            'disentangled_nonlinear', 'disentangled_patch', 'tgn', 'volterra'])
    p.add_argument('--traj_model', type=str, default=DEFAULTS['traj_model'],
                   choices=['baller2vec', 'star', 'lstm', 'constant_velocity', 'vrnn'])
    p.add_argument('--use_context', action='store_true', default=True)
    p.add_argument('--no_context', dest='use_context', action='store_false')
    
    # Sparse volterra
    p.add_argument('--mp_layers', type=int, default=2)
    p.add_argument('--topk', type=int, default=4)
    p.add_argument('--sparse_volterra_Q', type=int, default=4)
    p.add_argument('--ffn_mult', type=int, default=2)

    # Voronoi spatial
    p.add_argument('--use_voronoi_spatial', action='store_true', default=False)
    p.add_argument('--voronoi_image_H', type=int, default=32)
    p.add_argument('--voronoi_image_W', type=int, default=48)
    p.add_argument('--voronoi_window_K', type=int, default=8)
    p.add_argument('--voronoi_stride', type=int, default=None)
    p.add_argument('--use_event_emb', action='store_true', default=False)
    
    p.add_argument('--ortho_weight', type=float, default=1.0)
    p.add_argument('--align_weight', type=float, default=1.0)
    p.add_argument('--align_type', type=str, default='cosine', choices=['cosine', 'infonce'])
    p.add_argument('--w_disentangle', type=float, default=0.1)

    # ✅ Curriculum learning (删除旧参数，添加新参数)
    p.add_argument('--curriculum_type', type=str, default=DEFAULTS['curriculum_type'],
                   choices=['step', 'linear', 'cosine', 'exponential', 'multistage'])
    
    # 1. 轨迹相关
    p.add_argument('--traj_short_only_epochs', type=int, default=DEFAULTS['traj_short_only_epochs'],
                   help='Epochs to train only on short trajectory')
    p.add_argument('--traj_warmup_epochs', type=int, default=DEFAULTS['traj_warmup_epochs'],
                   help='Epochs to transition from short to full trajectory')
                   
    # 2. 事件相关
    p.add_argument('--event_start_epoch', type=int, default=DEFAULTS['event_start_epoch'],
                   help='Epoch to start training event prediction')
    p.add_argument('--event_warmup_epochs', type=int, default=DEFAULTS['event_warmup_epochs'],
                   help='Epochs to warm up event weights from 0 to final')

    # 3. 权重范围
    p.add_argument('--initial_event_weight', type=float, default=DEFAULTS['initial_event_weight'])
    p.add_argument('--final_event_weight', type=float, default=DEFAULTS['final_event_weight'])
    p.add_argument('--initial_disentangle_weight', type=float, default=DEFAULTS['initial_disentangle_weight'])
    p.add_argument('--final_disentangle_weight', type=float, default=DEFAULTS['final_disentangle_weight'])
    
    return p.parse_args()


def _distributed_requested(args) -> bool:
    return bool(args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _strip_module_prefix(state_dict):
    if state_dict and all(str(k).startswith("module.") for k in state_dict.keys()):
        return {str(k)[7:]: v for k, v in state_dict.items()}
    return state_dict


def _limit_files(files, limit):
    if limit is None or limit <= 0:
        return files
    return files[:limit]


def _require_existing_data_paths(args):
    missing = []
    if not args.pt_dir:
        missing.append("--pt_dir")
    if not args.sample_ml_dir:
        missing.append("--sample_ml_dir")
    if missing:
        joined = " and ".join(missing)
        raise ValueError(
            f"{joined} must be provided. See README.md for expected dataloader "
            "and sample metadata directories."
        )
    if not os.path.isdir(args.pt_dir):
        raise FileNotFoundError(f"--pt_dir does not exist: {args.pt_dir}")
    if not os.path.isdir(args.sample_ml_dir):
        raise FileNotFoundError(f"--sample_ml_dir does not exist: {args.sample_ml_dir}")


def _dummy_labels(cfg):
    labels = [f"h{i}" for i in range(1, cfg.n_home + 1)]
    labels += [f"a{i}" for i in range(1, cfg.n_away + 1)]
    labels = labels[:cfg.n_agents]
    if len(labels) < cfg.n_agents:
        labels += [f"PAD_h{i}" for i in range(cfg.n_agents - len(labels))]
    return labels


def _materialize_lazy_modules(model, cfg, F_dyn, F_edge, dyn_cols, edge_cols, device):
    real = _unwrap(model)
    if getattr(real.context_encoder, "player_token_proj", None) is not None:
        return
    C = max(int(getattr(cfg, "C", 1)), 1)
    labels = _dummy_labels(cfg)
    labels_batch = [[labels for _ in range(C)]]
    dummy_dyn = torch.zeros(1, C, F_dyn, device=device)
    dummy_edge = torch.zeros(1, C, F_edge, device=device)
    was_training = real.training
    real.eval()
    with torch.no_grad():
        real.context_encoder(
            dummy_dyn,
            dummy_edge,
            labels_batch,
            cols=dyn_cols,
            edge_cols=edge_cols,
            return_team_labels=True,
        )
    if was_training:
        real.train()


def _load_checkpoint(model, checkpoint_path, device, strict=False, rank=0):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("head", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state = _strip_module_prefix(state)
    incompatible = model.load_state_dict(state, strict=strict)
    if rank == 0:
        epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
        print(f"[checkpoint] loaded {checkpoint_path} (epoch={epoch})")
        if not strict:
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if missing:
                print(f"[checkpoint] missing keys: {len(missing)}")
            if unexpected:
                print(f"[checkpoint] unexpected keys: {len(unexpected)}")
    return checkpoint


def _print_eval_summary(name, metrics):
    ade_short = metrics["ADE_short"]
    fde_short = metrics["FDE_short"]
    ade = metrics["ADE"]
    fde = metrics["FDE"]
    acc1 = metrics["acc_tgt"]
    acc3 = metrics["acc_tgt_top3"]
    acc5 = metrics["acc_tgt_top5"]
    mae1 = metrics["mae_eidx"]
    mae3 = metrics["mae_eidx_top3"]
    mae5 = metrics["mae_eidx_top5"]
    pos1 = metrics["mse_epos"]
    pos3 = metrics["mse_epos_top3"]
    pos5 = metrics["mse_epos_top5"]
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY - {name}")
    print("=" * 80)
    print(f"Trajectory short: ADE={ade_short:.4f}, FDE={fde_short:.4f}")
    print(f"Trajectory long:  ADE={ade:.4f}, FDE={fde:.4f}")
    print(f"Target player:    Top1={acc1:.4f}, Top3={acc3:.4f}, Top5={acc5:.4f}")
    print(f"Event time MAE:   Top1={mae1:.4f}, Top3={mae3:.4f}, Top5={mae5:.4f}")
    print(f"Event pos MSE:    Top1={pos1:.4f}, Top3={pos3:.4f}, Top5={pos5:.4f}")
    print("=" * 80)

def main():
    args = parse_args()

    _require_existing_data_paths(args)
    args.output_dir = os.path.abspath(args.output_dir)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    runs_dir = os.path.join(args.output_dir, "runs")
    metrics_dir = os.path.join(args.output_dir, "metrics")

    # =============== DDP initialization ===============
    distributed = _distributed_requested(args)
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=30))
        rank = get_rank()
        world_size = get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available() and args.device != "cpu":
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(args.device if str(args.device).startswith("cuda") else "cuda")
    else:
        device = torch.device("cpu")

    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    seed_base = 42
    random.seed(seed_base + rank)
    np.random.seed(seed_base + rank)
    torch.manual_seed(seed_base + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base + rank)

    # =============== Configuration ===============
    class Cfg:
        pass
    cfg = Cfg()
    for k, v in DEFAULTS.items():
        setattr(cfg, k, v)

    cfg.short_fut_T = args.short_fut_T  
    cfg.w_short_traj = args.w_short_traj  
    cfg.dynamic_weight = args.dynamic_weight

    cfg.device = str(device)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.C = args.C
    cfg.H = args.H
    cfg.n_agents = int(args.n_agents)
    cfg.n_home = int(args.n_home) if args.n_home is not None else cfg.n_agents // 2
    cfg.n_away = int(args.n_away) if args.n_away is not None else cfg.n_agents - cfg.n_home
    if cfg.n_home + cfg.n_away != cfg.n_agents:
        raise ValueError(f"n_home + n_away must equal n_agents, got {cfg.n_home} + {cfg.n_away} != {cfg.n_agents}")
    cfg.ball_idx = int(args.ball_idx) if args.ball_idx is not None else cfg.n_agents
    cfg.n_total_agents = int(max(cfg.n_agents + 1, cfg.ball_idx + 1))
    cfg.field_L = float(args.field_L)
    cfg.field_W = float(args.field_W)
    cfg.max_side_id = int(max(args.max_side_id, cfg.n_home, cfg.n_away))
    cfg.disable_dyn_ctx = args.disable_dyn_ctx
    cfg.disable_edge_ctx = args.disable_edge_ctx
    cfg.disable_prior = args.disable_prior
    cfg.volterra_Q = args.volterra_Q
    cfg.volterra_scales = args.volterra_scales
    cfg.traj_model = args.traj_model.lower()
    cfg.use_context = args.use_context
    cfg.event_head_type = args.event_head.lower()
    cfg.predict_ball = args.predict_ball
    cfg.w_ball = args.w_ball
    cfg.use_event_emb = args.use_event_emb
    cfg.use_voronoi_spatial = args.use_voronoi_spatial
    cfg.ortho_weight = args.ortho_weight
    cfg.align_weight = args.align_weight
    cfg.align_type = args.align_type
    cfg.w_disentangle = args.w_disentangle
    cfg.n_event_types = int(args.n_event_types)
    cfg.output_dir = args.output_dir

    LMU_D = cfg.LMU_D
    F_LMU = cfg.F_LMU
    LMU_H = cfg.LMU_H

    if rank == 0:
        print("=" * 60)
        print(f"🎯 Trajectory Model: {cfg.traj_model.upper()}")
        print(f"🎯 Event Head: {cfg.event_head_type.upper()}")
        print(f"🎯 Use Context: {cfg.use_context}")
        print(f"🎯 Agents: {cfg.n_agents} ({cfg.n_home}+{cfg.n_away}), ball_idx={cfg.ball_idx}, field={cfg.field_L:g}x{cfg.field_W:g}")

        if cfg.event_head_type == 'disentangled':
            print(f"🎯 Dynamic Weight: {cfg.dynamic_weight:.2f}")
            if cfg.dynamic_weight == 0.0:
                print(f"   → Intent Only (context-based features)")
            elif cfg.dynamic_weight == 1.0:
                print(f"   → Dynamic Only (trajectory-based features)")
            else:
                intent_pct = (1.0 - cfg.dynamic_weight) * 100
                dynamic_pct = cfg.dynamic_weight * 100
                print(f"   → Balanced ({intent_pct:.0f}% intent + {dynamic_pct:.0f}% dynamic)")

        elif cfg.event_head_type == 'disentangled_gated':
            print(f"🎯 Fusion Type: Automatic Gated Fusion")
            print(f"   → Gate network learns dynamic weights per player")

        elif cfg.event_head_type == 'disentangled_nonlinear':
            print(f"🎯 Fusion Type: Nonlinear Fusion")
            print(f"   → MLP-based nonlinear fusion of intent and dynamic features")

        print(f"🎯 Volterra Q={cfg.volterra_Q}, scales={cfg.volterra_scales}")
        print(f"🎯 Games: {args.games}")
        print("=" * 60)

    # =============== Load column names ===============
    edge_npz = np.load(os.path.join(args.sample_ml_dir, "edges_raw_2d.npz"), allow_pickle=True)
    dyn_npz = np.load(os.path.join(args.sample_ml_dir, "nodes_dynamic_2d.npz"), allow_pickle=True)

    def _norm(s):
        return s.decode('utf-8') if isinstance(s, (bytes, np.bytes_)) else str(s)

    edge_cols = np.array([_norm(c) for c in edge_npz["columns"]], dtype=object)
    dyn_cols = np.array([_norm(c) for c in dyn_npz["columns"]], dtype=object)

    F_edge = len(edge_cols)
    F_dyn = len(dyn_cols)

    if rank == 0:
        print(f"Edge columns: {F_edge}, Dyn columns: {F_dyn}")

    # =============== Data loading ===============
    t0 = time.time()
    train_files = sorted(glob(os.path.join(args.pt_dir, "train", "**", "*.pt"), recursive=True))
    val_files = sorted(glob(os.path.join(args.pt_dir, "val", "**", "*.pt"), recursive=True))
    test_files = sorted(glob(os.path.join(args.pt_dir, "test", "**", "*.pt"), recursive=True))
    full_counts = (len(train_files), len(val_files), len(test_files))
    train_files = _limit_files(train_files, args.limit_train_files)
    val_files = _limit_files(val_files, args.limit_val_files)
    test_files = _limit_files(test_files, args.limit_test_files)

    if rank == 0:
        print(f"Found {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")
        if full_counts != (len(train_files), len(val_files), len(test_files)):
            print(
                "Using limited file counts for smoke/debug run "
                f"(full train/val/test={full_counts[0]}/{full_counts[1]}/{full_counts[2]})"
            )

    if args.eval_only:
        if not test_files:
            raise RuntimeError(f"No test .pt files found under {args.pt_dir}")
        train_ds = None
        val_ds = None
        test_ds = MatchDataset(test_files, debug=False, rank=rank)
    else:
        if not train_files or not val_files or not test_files:
            raise RuntimeError(
                f"Expected train/val/test .pt files under {args.pt_dir}, got "
                f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
            )
        train_ds = MatchDataset(train_files, debug=False, rank=rank)
        val_ds = MatchDataset(val_files, debug=False, rank=rank)
        test_ds = MatchDataset(test_files, debug=False, rank=rank)
    
    if rank == 0:
        print(f"Dataset init done in {time.time()-t0:.2f}s")

    collate_adapt = make_adaptive_collate(target_bs=cfg.batch_size, random_pick=False)

    if distributed:
        train_sampler = None if train_ds is None else DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = None if val_ds is None else DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = None if train_ds is None else DataLoader(
        train_ds, batch_size=1, sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=collate_adapt, persistent_workers=(args.num_workers > 0)
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds, batch_size=1, sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=collate_adapt, persistent_workers=(args.num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, sampler=test_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=collate_adapt, persistent_workers=(args.num_workers > 0)
    )

    # =============== Model construction ===============
    ctx_enc = ContextEncoder(d_dyn=F_dyn, d_edge=F_edge, d_h=cfg.h_dim, attn_pool=False, n_home=cfg.n_home, n_away=cfg.n_away, max_side_id=cfg.max_side_id)
    prior = None if args.disable_prior else TgtPlayerScorer(field_L=cfg.field_L, field_W=cfg.field_W, use_voronoi=True, n_players=cfg.n_agents)

    parallel_head = ParallelEventPredictor_Unified(
        context_encoder=ctx_enc,
        prior_scorer=prior,
        hidden_dim=cfg.h_dim,
        n_players=cfg.n_agents,
        fut_T=cfg.fut_T,
        n_event_types=cfg.n_event_types,
        d_model=128,
        dropout=0.1,
        event_head_type=cfg.event_head_type,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        volterra_Q=cfg.volterra_Q,
        volterra_scales=cfg.volterra_scales,
        mp_layers=args.mp_layers,
        topk=args.topk,
        sparse_volterra_Q=args.sparse_volterra_Q,
        ffn_mult=args.ffn_mult,
        dynamic_weight=cfg.dynamic_weight,
        use_gated_fusion=(cfg.event_head_type == 'disentangled_gated'),
        use_hsic=args.use_hsic, 
        decay_rate=args.decay_rate,
    )

    context_dim = cfg.h_dim
    if cfg.predict_ball:
        parallel_head.ball_step_head = BallStepHead(
            hist_len=cfg.burn_in, d_h=LMU_H, d_ctx=context_dim, d_attn=128,
        )

    # Import model classes
    if cfg.traj_model == 'baller2vec':
        if cfg.use_context:
            from model_traj.baller2vec import (
                Baller2VecWindowedWithContext as AgentLMU,
                TransformerTrajStepHeadWithContext as TrajStepHead
            )
        else:
            from model_traj.baller2vec import (
                Baller2VecWindowedNoContext as AgentLMU,
                TransformerTrajStepHeadNoContext as TrajStepHead
            )
    elif cfg.traj_model == 'star':
        if cfg.use_context:
            from model_traj.star import (
                STARWindowedWithContext as AgentLMU,
                TransformerTrajStepHeadWithContext as TrajStepHead
            )
        else:
            from model_traj.star import (
                STARWindowedNoContext as AgentLMU,
                TransformerTrajStepHeadNoContext as TrajStepHead
            )
    elif cfg.traj_model == 'lstm':
        from model_traj.lstm import (
            AgentLSTM as AgentLMU,
            LSTMTrajStepHead as TrajStepHead
        )
    elif cfg.traj_model == 'constant_velocity':
            from model_traj.constant_velocity import (
            ConstantVelocityPredictor as AgentLMU,
            ConstantVelocityTrajStepHead as TrajStepHead
        )
            
    elif cfg.traj_model == 'vrnn':
        if cfg.use_context:
            from model_traj.vrnn import (
                AgentVRNNWithContext as AgentLMU,
                VRNNTrajStepHeadWithContext as TrajStepHead
            )
        else:
            from model_traj.vrnn import (
                AgentVRNN as AgentLMU,
                VRNNTrajStepHead as TrajStepHead
            )
    else:
        raise ValueError(f"Unknown traj_model: {cfg.traj_model}")

    parallel_head.lmu_proj_agent = torch.nn.Linear(2, F_LMU)

    # Configure agent_lmu
    if cfg.traj_model == 'baller2vec':
        if cfg.use_context:
            parallel_head.agent_lmu = AgentLMU(
                in_dim=F_LMU, h_dim=LMU_H, window_K=8, d_mem=None,
                n_heads=4, n_layers=2, dropout=0.1,
                context_dim=context_dim, n_players=cfg.n_agents, use_event_as_cls=False,
            )
        else:
            parallel_head.agent_lmu = AgentLMU(
                in_dim=F_LMU, h_dim=LMU_H, window_K=8, d_mem=None,
                n_heads=4, n_layers=2, dropout=0.1, n_players=cfg.n_agents,
            )
    elif cfg.traj_model == 'star':
        if cfg.use_context:
            parallel_head.agent_lmu = AgentLMU(
                in_dim=F_LMU, h_dim=LMU_H, window_K=8, d_mem=None,
                n_heads=4, n_layers=2, dropout=0.1,
                context_dim=context_dim, n_players=cfg.n_agents,
            )
        else:
            parallel_head.agent_lmu = AgentLMU(
                in_dim=F_LMU, h_dim=LMU_H, window_K=8, d_mem=None,
                n_heads=4, n_layers=2, dropout=0.1, n_players=cfg.n_agents,
            )
    elif cfg.traj_model == 'lstm':
        parallel_head.agent_lmu = AgentLMU(in_dim=F_LMU, h_dim=LMU_H)
    elif cfg.traj_model == 'constant_velocity':
        parallel_head.agent_lmu = AgentLMU(
            in_dim=F_LMU, 
            h_dim=LMU_H,
            velocity_window=5,        # 使用最后5帧
            use_acceleration=False,
        )
    elif cfg.traj_model == 'vrnn':
        if cfg.use_context:
            parallel_head.agent_lmu = AgentLMU(
                in_dim=F_LMU,
                h_dim=LMU_H,
                z_dim=32,  # VRNN latent dimension
                window_K=8,
                d_mem=LMU_D,
                context_dim=context_dim,
                n_players=cfg.n_agents,
                dropout=0.1,
            )
        else:
            parallel_head.agent_lmu = AgentLMU(
                in_dim=F_LMU,
                h_dim=LMU_H,
                z_dim=32,
                window_K=8,
                d_mem=LMU_D,
                n_players=cfg.n_agents,
                dropout=0.1,
            ) 
    else:
        raise ValueError(f"Unknown traj_model: {cfg.traj_model}")

    # Trajectory step head
    if cfg.traj_model == 'lstm':
        parallel_head.traj_step_head = TrajStepHead(
            in_dim=LMU_H, hidden=256, dropout=0.1,
            use_m=cfg.use_context,
            m_dim=LMU_H if cfg.use_context else None,
        )
    elif cfg.traj_model == 'constant_velocity':
        # ✅ 添加这个分支
        parallel_head.traj_step_head = TrajStepHead(
            in_dim=LMU_H,
            hidden=256,
            dropout=0.1,
            use_m=True,
            m_dim=LMU_H,
        )

    elif cfg.traj_model == 'vrnn':
        if cfg.use_context:
            parallel_head.traj_step_head = TrajStepHead(
                in_dim=LMU_H,
                hidden=256,
                dropout=0.1,
                use_m=False,
                m_dim=None,
                context_dim=context_dim,
            )
        else:
            parallel_head.traj_step_head = TrajStepHead(
                in_dim=LMU_H,
                hidden=256,
                dropout=0.1,
                use_m=False,
                m_dim=None,
            )

    else:
        use_m = False if cfg.traj_model in ['baller2vec', 'star'] else True
        m_dim = None if cfg.traj_model in ['baller2vec', 'star'] else LMU_D
        
        traj_head_kwargs = {
            'in_dim': LMU_H,
            'hidden': 256,
            'dropout': 0.1,
            'use_m': use_m,
            'm_dim': m_dim,
        }
        
        if cfg.use_context:
            traj_head_kwargs.update({'context_dim': context_dim})
        
        parallel_head.traj_step_head = TrajStepHead(**traj_head_kwargs)
    
    parallel_head.lmu_gamma = 0.9

    parallel_head = parallel_head.to(device)
    _materialize_lazy_modules(parallel_head, cfg, F_dyn, F_edge, dyn_cols, edge_cols, device)
    if args.resume:
        _load_checkpoint(parallel_head, args.resume, device, strict=args.strict_resume, rank=rank)

    if distributed:
        head = DDP(
            parallel_head,
            device_ids=[local_rank] if torch.cuda.is_available() and device.type == "cuda" else None,
            output_device=local_rank if torch.cuda.is_available() and device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    else:
        head = parallel_head

    optimizer = torch.optim.Adam(list(head.parameters()), lr=cfg.lr)

    # =============== Curriculum Scheduler ===============
    # if args.curriculum_type == 'multistage':
    #     curriculum_scheduler = MultiStageScheduler(
    #         stage_epochs=[args.traj_only_epochs, args.warmup_epochs, 10, None],
    #         total_epochs=cfg.epochs,
    #     )
    #     if rank == 0:
    #         print("=" * 60)
    #         print("🎓 Using MultiStage Curriculum Learning")
    #         print(f"   Stage boundaries: {curriculum_scheduler.stage_boundaries}")
    #         print("=" * 60)
    # else:
    # ✅ 使用新的解耦参数初始化
    curriculum_scheduler = CurriculumScheduler(
        # 1. Trajectory Timeline
        traj_short_only_epochs=args.traj_short_only_epochs,
        traj_warmup_epochs=args.traj_warmup_epochs,
        
        # 2. Event Timeline
        event_start_epoch=args.event_start_epoch,
        event_warmup_epochs=args.event_warmup_epochs,
        
        # 3. Others
        schedule_type=args.curriculum_type,
        initial_event_weight=args.initial_event_weight,
        final_event_weight=args.final_event_weight,
        initial_disentangle_weight=args.initial_disentangle_weight,
        final_disentangle_weight=args.final_disentangle_weight,
        
        tgt_weight_scale=cfg.w_tgt,
        eidx_weight_scale=cfg.w_eidx,
        epos_weight_scale=cfg.w_epos,
        
        initial_short_weight=args.initial_short_weight,
        final_short_weight=args.final_short_weight,
    )
    
    if rank == 0:
        print("=" * 60)
        print(f"🎓 Curriculum Learning: {args.curriculum_type}")
        print(curriculum_scheduler) # 这会打印新的 str 信息
        print("=" * 60)

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval_only requires --resume/--checkpoint")
        val_weights_full = {
            "traj": 1.0,
            "tgt": cfg.w_tgt,
            "eidx": cfg.w_eidx,
            "epos": cfg.w_epos,
            "disentangle": cfg.w_disentangle,
            "phase": "full",
        }
        test_result = evaluate_curriculum(
            test_loader,
            head,
            cfg,
            mode="test",
            curriculum_weights=val_weights_full,
            dyn_cols=dyn_cols,
            edge_cols=edge_cols,
        )
        if rank == 0:
            os.makedirs(metrics_dir, exist_ok=True)
            eval_name = args.run_name or f"{cfg.traj_model}_{cfg.event_head_type}_eval"
            _print_eval_summary(eval_name, test_result)
            metrics_path = os.path.join(metrics_dir, f"eval_{eval_name}_{args.games}games.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_result, f, indent=2)
            print(f"[eval] saved metrics to {metrics_path}")
        return

    # Training setup 
    best_val = float('inf')
    # =============== 动态构建唯一模型名称 (修复重名问题) ===============
    # 基础模型名
    model_name = f"{cfg.traj_model}_{cfg.event_head_type}"

    # 1. 针对 disentangled_patch 分支，加入 HSIC 和 Dynamic Weight 标签
    if cfg.event_head_type == 'disentangled_patch':
        hsic_tag = "HSIC" if args.use_hsic else "NoHSIC"
        dw_tag = f"dw{str(cfg.dynamic_weight).replace('.', '')}"
        model_name += f"_{hsic_tag}_{dw_tag}"
    
    # 2. 针对原有 disentangled 分支的兼容
    elif cfg.event_head_type == 'disentangled':
        dw_tag = f"dw{str(cfg.dynamic_weight).replace('.', '')}"
        model_name += f"_{dw_tag}"

    # 3. 加入 Context 标签
    model_name += f"_{'ctx' if cfg.use_context else 'noctx'}"
    
    # 4. 加入课程学习标识
    model_name += "_curriculum"
    if args.run_name:
        model_name = args.run_name

    # =============================================================

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoints_dir, f"best_{model_name}_{args.games}games.pt")
    best_ckpt_path_combo = os.path.join(checkpoints_dir, f"best_{model_name}_combo_{args.games}games.pt")

    epoch_logs = []
    early_stopper = EarlyStopping(patience=8, delta=1e-4)
    early_stop_epoch = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    
    base = None
    best_val_metric = float('inf')

    writer = None
    if rank == 0:
        run_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(runs_dir, f"{model_name}_{args.games}games_{run_tag}")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[TB] log_dir = {log_dir}")

    # =============== Training Loop ===============
    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Get curriculum weights
        curriculum_weights = curriculum_scheduler.get_weights(epoch)
        if rank == 0:
            print(f"\n[Epoch {epoch}] Phase: {curriculum_weights['phase']}")
            print(f"  Trajectory: w_short={curriculum_weights['w_short_traj']:.2f}, "  # ✅ NEW
                f"w_long={1.0 - curriculum_weights['w_short_traj']:.2f}")
            print(f"  Events: tgt={curriculum_weights['tgt']:.2f}, "
                f"eidx={curriculum_weights['eidx']:.2f}, "
                f"disentangle={curriculum_weights['disentangle']:.4f}")
            

        SUM_KEYS = [
            'total_sum', 'nll_sum', 'kld_sum',
            'traj_loss_short_sum', 'ADE_short_sum', 'FDE_short_sum',  # ✅ NEW: Short-term
            'traj_loss_sum', 'ADE_sum', 'FDE_sum', 'ball_loss_sum',
            'tgt_loss_sum', 'eidx_loss_sum', 'epos_loss_sum',
            'acc_tgt_sum', 'mae_eidx_sum', 'mse_epos_sum',
            'acc_tgt_top3_sum', 'mae_eidx_top3_sum', 'mse_epos_top3_sum',
            'acc_tgt_top5_sum', 'mae_eidx_top5_sum', 'mse_epos_top5_sum',
            'disentangle_loss_sum', 'ortho_loss_sum', 'align_loss_sum',
            'adv_loss_sum', 'disc_acc_sum', 'hsic_score_sum', 
            'aux_epos_tf_sum', 'aux_win_sum',
        ]
        COUNT_KEYS = ['count_total', 'count_valid']

        sums = {k: 0.0 for k in SUM_KEYS}
        counts = {k: 0.0 for k in COUNT_KEYS}

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch:03d} [rank {rank}]",
                    dynamic_ncols=True, mininterval=1.0, disable=(rank != 0))

        for i, batch in enumerate(pbar):
            has_none = torch.tensor([1 if batch is None else 0], device=device)
            if is_dist_avail_and_initialized():
                dist.all_reduce(has_none, op=dist.ReduceOp.MAX)

            if has_none.item() > 0:
                if rank == 0:
                    tqdm.write(f"[WARN] Batch {i}: skip (None)")
                logs = {k: 0.0 for k in SUM_KEYS + COUNT_KEYS}
            else:
                logs = step_batch_curriculum(
                    batch, head, cfg, 
                    mode='train', 
                    optimizer=optimizer, 
                    curriculum_weights=curriculum_weights,
                    dyn_cols=dyn_cols,
                    edge_cols=edge_cols
                )

            for k in SUM_KEYS:
                sums[k] += float(logs.get(k, 0.0))
            for k in COUNT_KEYS:
                counts[k] += float(logs.get(k, 0.0))

        # Aggregate across ranks
        sums_vec = torch.tensor([sums[k] for k in SUM_KEYS], device=device)
        counts_vec = torch.tensor([counts[k] for k in COUNT_KEYS], device=device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(sums_vec, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts_vec, op=dist.ReduceOp.SUM)

        sums = {k: float(v) for k, v in zip(SUM_KEYS, sums_vec.tolist())}
        counts = {k: float(v) for k, v in zip(COUNT_KEYS, counts_vec.tolist())}

        ct = max(counts['count_total'], 1.0)
        cv = max(counts['count_valid'], 1.0)

        train_avg = {}
        for src, dst in [
            ('total_sum', 'total'), ('nll_sum', 'nll'), ('kld_sum', 'kld'),
            ('traj_loss_short_sum', 'traj_loss_short'),  # ✅ NEW
            ('ADE_short_sum', 'ADE_short'),  # ✅ NEW
            ('FDE_short_sum', 'FDE_short'),  # ✅ NEW
            ('traj_loss_sum', 'traj_loss'), ('ADE_sum', 'ADE'), ('FDE_sum', 'FDE'),
            ('ball_loss_sum', 'ball_loss'),
            ('disentangle_loss_sum', 'disentangle_loss'),
            ('adv_loss_sum', 'adv_loss'),
            ('disc_acc_sum', 'disc_acc'),
            ('hsic_score_sum', 'hsic_score'),
            ('ortho_loss_sum', 'ortho_loss'),
            ('align_loss_sum', 'align_loss'),
        ]:
            train_avg[dst] = sums[src] / ct
        
        for src, dst in [
            ('tgt_loss_sum', 'tgt_loss'), ('eidx_loss_sum', 'eidx_loss'), ('epos_loss_sum', 'epos_loss'),
            ('acc_tgt_sum', 'acc_tgt'), ('mae_eidx_sum', 'mae_eidx'), ('mse_epos_sum', 'mse_epos'),
            ('acc_tgt_top3_sum', 'acc_tgt_top3'), ('mae_eidx_top3_sum', 'mae_eidx_top3'), 
            ('mse_epos_top3_sum', 'mse_epos_top3'),
            ('acc_tgt_top5_sum', 'acc_tgt_top5'), ('mae_eidx_top5_sum', 'mae_eidx_top5'), 
            ('mse_epos_top5_sum', 'mse_epos_top5'),
            ('aux_epos_tf_sum', 'aux_epos_tf'), ('aux_win_sum', 'aux_win'),
        ]:
            train_avg[dst] = sums[src] / cv

        if rank == 0:
            valid_ratio = (counts['count_valid'] / ct) * 100.0
            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"Total: {int(counts['count_total'])}, Valid: {int(counts['count_valid'])} ({valid_ratio:.1f}%)"
            )

        # =============== Validation ===============
        val_weights_full = {
            'traj': 1.0,
            'tgt': cfg.w_tgt,
            'eidx': cfg.w_eidx,
            'epos': cfg.w_epos,
            'disentangle': cfg.w_disentangle,
            'phase': 'full',
        }
        
        val_tf = evaluate_curriculum(val_loader, head, cfg, mode='val_tf', 
                                     curriculum_weights=val_weights_full,
                                     dyn_cols=dyn_cols, edge_cols=edge_cols)
        val_free = evaluate_curriculum(val_loader, head, cfg, mode='test',
                                       curriculum_weights=val_weights_full,
                                       dyn_cols=dyn_cols, edge_cols=edge_cols)

        # =============== Test ===============
        test_result = None
        should_test = False

        if args.test_every_n_epochs > 0:
            if (epoch + 1) % args.test_every_n_epochs == 0:
                should_test = True

        if epoch + 1 == cfg.epochs:
            should_test = True

        if should_test:
            if rank == 0:
                print(f"\n{'='*70}")
                print(f"🧪 Running Test at Epoch {epoch + 1}")
                print(f"{'='*70}")
            
            test_result = evaluate_curriculum(test_loader, head, cfg, mode='test',
                                            curriculum_weights=val_weights_full,
                                            dyn_cols=dyn_cols, edge_cols=edge_cols)
            
            if rank == 0:
                print(f"\n{'='*70}")
                print(f"📊 TEST RESULTS @ Epoch {epoch + 1}")
                print(f"{'='*70}")
                
                # ========== Trajectory Metrics ==========
                print(f"\n🎯 Trajectory Prediction:")
                print(f"  Short-term (2.8s @ {cfg.short_fut_T} frames):")
                print(f"    ADE:  {test_result['ADE_short']:.4f} m")
                print(f"    FDE:  {test_result['FDE_short']:.4f} m")
                print(f"  Long-term (6.4s @ {cfg.fut_T} frames):")
                print(f"    ADE:  {test_result['ADE']:.4f} m")
                print(f"    FDE:  {test_result['FDE']:.4f} m")
                
                # ========== Event - Target Player ==========
                print(f"\n🎯 Target Player Prediction:")
                print(f"  Top-1 Accuracy:  {test_result['acc_tgt']:.4f}")
                print(f"  Top-3 Accuracy:  {test_result['acc_tgt_top3']:.4f}")
                print(f"  Top-5 Accuracy:  {test_result['acc_tgt_top5']:.4f}")
                
                # ========== Event - Timing ==========
                print(f"\n⏱️  Event Timing Prediction (MAE in frames):")
                print(f"  Top-1 MAE:  {test_result['mae_eidx']:.2f} frames")
                print(f"  Top-3 MAE:  {test_result['mae_eidx_top3']:.2f} frames")
                print(f"  Top-5 MAE:  {test_result['mae_eidx_top5']:.2f} frames")
                
                # ========== Event - Position ==========
                print(f"\n📍 Event Position Prediction (MSE in m²):")
                print(f"  Top-1 MSE:  {test_result['mse_epos']:.4f} m²")
                print(f"  Top-3 MSE:  {test_result['mse_epos_top3']:.4f} m²")
                print(f"  Top-5 MSE:  {test_result['mse_epos_top5']:.4f} m²")
                
                # ========== Disentanglement (if applicable) ==========
                if 'ortho_loss' in test_result:
                    print(f"\n🔀 Disentanglement Metrics:")
                    print(f"  Orthogonality:  {test_result.get('ortho_loss', 0.0):.4f}")
                    print(f"  Alignment:      {test_result.get('align_loss', 0.0):.4f}")
                
                print(f"{'='*70}\n")

        # Combined metric + early stopping
        if rank == 0:
            scheduler.step(val_free['ADE'])
            
            # ========== Trajectory Metrics ==========
            tqdm.write(
                f"Epoch {epoch:03d} TRAJECTORY:\n"
                f"  Train: Short ADE={train_avg['ADE_short']:.3f}, FDE={train_avg['FDE_short']:.3f} | "
                f"Long ADE={train_avg['ADE']:.3f}, FDE={train_avg['FDE']:.3f}\n"
                f"  Val:   Short ADE={val_free['ADE_short']:.3f}, FDE={val_free['FDE_short']:.3f} | "
                f"Long ADE={val_free['ADE']:.3f}, FDE={val_free['FDE']:.3f}"
            )
            
            # ========== Event Metrics ==========
            tqdm.write(
                f"Epoch {epoch:03d} EVENTS:\n"
                f"  Target:  Top1={val_free['acc_tgt']:.3f}, Top3={val_free['acc_tgt_top3']:.3f}, "
                f"Top5={val_free['acc_tgt_top5']:.3f}\n"
                f"  Timing:  Top1={val_free['mae_eidx']:.2f}, Top3={val_free['mae_eidx_top3']:.2f}, "
                f"Top5={val_free['mae_eidx_top5']:.2f} frames\n"
                f"  Position: Top1={val_free['mse_epos']:.4f}, Top3={val_free['mse_epos_top3']:.4f}, "
                f"Top5={val_free['mse_epos_top5']:.4f} m²"
            )

            if base is None:
                base = {
                    'ADE': max(float(val_free['ADE']), 1e-6),
                    'acc_tgt': min(max(float(val_free['acc_tgt']), 1e-6), 0.9999),
                    'mae_eidx': max(float(val_free['mae_eidx']), 1e-6),
                    'mse_epos': max(float(val_free['mse_epos']), 1e-6),
                }

            lam1, lam2, lam3 = 1.0, 0.5, 0.5
            denom_acc = max(1.0 - base['acc_tgt'], 1e-6)
            val_metric = (
                (float(val_free['ADE']) / base['ADE']) +
                lam1 * ((1.0 - float(val_free['acc_tgt'])) / denom_acc) +
                lam2 * (float(val_free['mae_eidx']) / base['mae_eidx']) +
                lam3 * ((float(val_free['mse_epos']) ** 0.5) / (base['mse_epos'] ** 0.5))
            )
            
            _ = early_stopper.step(val_metric)
            
            if early_stopper.early_stop and early_stop_epoch is None:
                early_stop_epoch = epoch
                tqdm.write("=" * 70)
                tqdm.write(f"⚠️  EARLY STOPPING at Epoch {epoch}")
                tqdm.write(f"   Best combo: {early_stopper.best_score:.5f}")
                tqdm.write("=" * 70)
                
                early_stop_ckpt = best_ckpt_path_combo.replace('.pt', '_earlystop.pt')
                os.makedirs(checkpoints_dir, exist_ok=True)
                torch.save({
                    'ctx_enc': _unwrap(head).context_encoder.state_dict(),
                    'head': _unwrap(head).state_dict(),
                    'epoch': epoch,
                    'combo_metric': val_metric,
                    'val_ADE': float(val_free['ADE']),
                    'config': {'traj_model': cfg.traj_model, 'use_context': cfg.use_context}
                }, early_stop_ckpt)
                tqdm.write(f"💾 Saved -> {early_stop_ckpt}\n")

            if val_metric < best_val_metric:
                best_val_metric = val_metric
                os.makedirs(checkpoints_dir, exist_ok=True)
                torch.save({
                    'ctx_enc': _unwrap(head).context_encoder.state_dict(),
                    'head': _unwrap(head).state_dict(),
                    'epoch': epoch,
                    'combo_metric': best_val_metric,
                    'val_ADE': float(val_free['ADE']),
                    'config': {'traj_model': cfg.traj_model, 'use_context': cfg.use_context}
                }, best_ckpt_path_combo)
                tqdm.write(f"✅ Best combo -> {best_ckpt_path_combo}")

            if val_free['ADE'] < best_val:
                best_val = val_free['ADE']
                os.makedirs(checkpoints_dir, exist_ok=True)
                torch.save({
                    'ctx_enc': _unwrap(head).context_encoder.state_dict(),
                    'head': _unwrap(head).state_dict(),
                    'epoch': epoch,
                    'val_ADE': best_val,
                    'config': {'traj_model': cfg.traj_model, 'use_context': cfg.use_context}
                }, best_ckpt_path)
                tqdm.write(f"✅ Best ADE -> {best_ckpt_path}")

      # =============== TensorBoard ===============
        if rank == 0 and writer is not None:
            # ========== Curriculum Progress ==========
            writer.add_scalar("curriculum/phase_id",
                                    ["Short_Traj", "Traj_Warmup", "Full_Traj", "Event_Warmup", "Fully_Trained"].index(curriculum_weights['phase']), epoch)
            writer.add_scalar("curriculum/event_weight", curriculum_weights['tgt'], epoch)
            writer.add_scalar("curriculum/disentangle_weight", curriculum_weights['disentangle'], epoch)
            writer.add_scalar("curriculum/progress", curriculum_weights.get('progress', 1.0), epoch)
            writer.add_scalar("curriculum/short_traj_weight", curriculum_weights.get('w_short_traj', 0.5), epoch)
            writer.add_scalar("curriculum/progress_traj", curriculum_weights.get('traj_progress', 0.0), epoch) # ✅ 新增
            writer.add_scalar("curriculum/progress_event", curriculum_weights.get('event_progress', 0.0), epoch) # ✅ 新增

            # ========== Overall Metrics ==========
            writer.add_scalar("overall/total_loss", train_avg['total'], epoch)
            writer.add_scalar("overall/learning_rate", optimizer.param_groups[0]['lr'], epoch)
            
            # ========== Trajectory - Short-term (2.8s) ==========
            writer.add_scalar("trajectory/train/ADE_short", train_avg['ADE_short'], epoch)
            writer.add_scalar("trajectory/train/FDE_short", train_avg['FDE_short'], epoch)
            writer.add_scalar("trajectory/train/loss_short", train_avg['traj_loss_short'], epoch)
            
            writer.add_scalar("trajectory/val/ADE_short", val_free['ADE_short'], epoch)
            writer.add_scalar("trajectory/val/FDE_short", val_free['FDE_short'], epoch)
            
            if test_result:
                writer.add_scalar("trajectory/test/ADE_short", test_result['ADE_short'], epoch)
                writer.add_scalar("trajectory/test/FDE_short", test_result['FDE_short'], epoch)
            
            # ========== Trajectory - Long-term (6.4s) ==========
            writer.add_scalar("trajectory/train/ADE", train_avg['ADE'], epoch)
            writer.add_scalar("trajectory/train/FDE", train_avg['FDE'], epoch)
            writer.add_scalar("trajectory/train/loss", train_avg['traj_loss'], epoch)
            
            writer.add_scalar("trajectory/val/ADE", val_free['ADE'], epoch)
            writer.add_scalar("trajectory/val/FDE", val_free['FDE'], epoch)
            
            if test_result:
                writer.add_scalar("trajectory/test/ADE", test_result['ADE'], epoch)
                writer.add_scalar("trajectory/test/FDE", test_result['FDE'], epoch)
            
            # ========== Event - Target Player Accuracy ==========
            # Top-1
            writer.add_scalar("event_target/train/acc_top1", train_avg['acc_tgt'], epoch)
            writer.add_scalar("event_target/val/acc_top1", val_free.get('acc_tgt', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_target/test/acc_top1", test_result.get('acc_tgt', 0.0), epoch)
            
            # Top-3
            writer.add_scalar("event_target/train/acc_top3", train_avg['acc_tgt_top3'], epoch)
            writer.add_scalar("event_target/val/acc_top3", val_free.get('acc_tgt_top3', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_target/test/acc_top3", test_result.get('acc_tgt_top3', 0.0), epoch)
            
            # Top-5
            writer.add_scalar("event_target/train/acc_top5", train_avg['acc_tgt_top5'], epoch)
            writer.add_scalar("event_target/val/acc_top5", val_free.get('acc_tgt_top5', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_target/test/acc_top5", test_result.get('acc_tgt_top5', 0.0), epoch)
            
            # ========== Event - Timing MAE ==========
            # Top-1
            writer.add_scalar("event_timing/train/mae_top1", train_avg['mae_eidx'], epoch)
            writer.add_scalar("event_timing/val/mae_top1", val_free.get('mae_eidx', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_timing/test/mae_top1", test_result.get('mae_eidx', 0.0), epoch)
            
            # Top-3
            writer.add_scalar("event_timing/train/mae_top3", train_avg['mae_eidx_top3'], epoch)
            writer.add_scalar("event_timing/val/mae_top3", val_free.get('mae_eidx_top3', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_timing/test/mae_top3", test_result.get('mae_eidx_top3', 0.0), epoch)
            
            # Top-5
            writer.add_scalar("event_timing/train/mae_top5", train_avg['mae_eidx_top5'], epoch)
            writer.add_scalar("event_timing/val/mae_top5", val_free.get('mae_eidx_top5', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_timing/test/mae_top5", test_result.get('mae_eidx_top5', 0.0), epoch)
            
            # ========== Event - Position MSE ==========
            # Top-1
            writer.add_scalar("event_position/train/mse_top1", train_avg['mse_epos'], epoch)
            writer.add_scalar("event_position/val/mse_top1", val_free.get('mse_epos', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_position/test/mse_top1", test_result.get('mse_epos', 0.0), epoch)
            
            # Top-3
            writer.add_scalar("event_position/train/mse_top3", train_avg['mse_epos_top3'], epoch)
            writer.add_scalar("event_position/val/mse_top3", val_free.get('mse_epos_top3', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_position/test/mse_top3", test_result.get('mse_epos_top3', 0.0), epoch)
            
            # Top-5
            writer.add_scalar("event_position/train/mse_top5", train_avg['mse_epos_top5'], epoch)
            writer.add_scalar("event_position/val/mse_top5", val_free.get('mse_epos_top5', 0.0), epoch)
            if test_result:
                writer.add_scalar("event_position/test/mse_top5", test_result.get('mse_epos_top5', 0.0), epoch)
            
            # ========== Event - Loss Components ==========
            writer.add_scalar("event_loss/train/tgt_loss", train_avg['tgt_loss'], epoch)
            writer.add_scalar("event_loss/train/eidx_loss", train_avg['eidx_loss'], epoch)
            writer.add_scalar("event_loss/train/epos_loss", train_avg['epos_loss'], epoch)
            
            # ========== Disentanglement Metrics ==========
            writer.add_scalar("disentangle/train/total", train_avg['disentangle_loss'], epoch)
            writer.add_scalar("disentangle/train/adv_loss", train_avg.get('adv_loss', 0.0), epoch)
            writer.add_scalar("disentangle/train/disc_acc", train_avg.get('disc_acc', 0.5), epoch)
            writer.add_scalar("disentangle/train/hsic_score", train_avg.get('hsic_score', 0.0), epoch)
            writer.add_scalar("disentangle/train/ortho", train_avg['ortho_loss'], epoch)
            writer.add_scalar("disentangle/train/align", train_avg['align_loss'], epoch)
            
            writer.add_scalar("disentangle/val/total", val_free.get('disentangle_loss', 0.0), epoch)
            writer.add_scalar("disentangle/val/ortho", val_free.get('ortho_loss', 0.0), epoch)
            writer.add_scalar("disentangle/val/align", val_free.get('align_loss', 0.0), epoch)
            
            if test_result:
                writer.add_scalar("disentangle/test/total", test_result.get('disentangle_loss', 0.0), epoch)
                writer.add_scalar("disentangle/test/ortho", test_result.get('ortho_loss', 0.0), epoch)
                writer.add_scalar("disentangle/test/align", test_result.get('align_loss', 0.0), epoch)
            
            # ========== Ball Prediction (if enabled) ==========
            if cfg.predict_ball and 'ball_loss' in train_avg:
                writer.add_scalar("ball/train/loss", train_avg['ball_loss'], epoch)
        # =============== CSV logging ===============
        if rank == 0:
            epoch_log = {
                "epoch": epoch,
                "phase": curriculum_weights['phase'],
                "lr": optimizer.param_groups[0]['lr'],
                
                # Training
                "train_ADE": train_avg['ADE'],
                "train_ADE_short": train_avg['ADE_short'],
                "train_FDE": train_avg['FDE'],
                "train_FDE_short": train_avg['FDE_short'],
                
                # Validation
                "val_ADE": val_free['ADE'],
                "val_ADE_short": val_free['ADE_short'],
                "val_FDE": val_free['FDE'],
                "val_FDE_short": val_free['FDE_short'],
                "val_acc_tgt_top1": val_free['acc_tgt'],
                "val_acc_tgt_top3": val_free['acc_tgt_top3'],
                "val_acc_tgt_top5": val_free['acc_tgt_top5'],
                "val_mae_eidx_top1": val_free['mae_eidx'],
                "val_mae_eidx_top3": val_free['mae_eidx_top3'],
                "val_mae_eidx_top5": val_free['mae_eidx_top5'],
                
                # Best metrics
                "best_val_ADE": best_val,
                "best_combo": best_val_metric,
            }
            
            if test_result:
                epoch_log.update({
                    "test_ADE": test_result['ADE'],
                    "test_ADE_short": test_result['ADE_short'],
                    "test_FDE": test_result['FDE'],
                    "test_FDE_short": test_result['FDE_short'],
                    "test_acc_tgt_top1": test_result['acc_tgt'],
                    "test_acc_tgt_top3": test_result['acc_tgt_top3'],
                    "test_acc_tgt_top5": test_result['acc_tgt_top5'],
                    "test_mae_eidx_top1": test_result['mae_eidx'],
                    "test_mae_eidx_top3": test_result['mae_eidx_top3'],
                    "test_mae_eidx_top5": test_result['mae_eidx_top5'],
                    "test_mse_epos_top1": test_result['mse_epos'],
                    "test_mse_epos_top3": test_result['mse_epos_top3'],
                    "test_mse_epos_top5": test_result['mse_epos_top5'],
                })
            else:
                # Fill with empty strings when no test
                epoch_log.update({
                    "test_ADE": "", "test_ADE_short": "", "test_FDE": "", "test_FDE_short": "",
                    "test_acc_tgt_top1": "", "test_acc_tgt_top3": "", "test_acc_tgt_top5": "",
                    "test_mae_eidx_top1": "", "test_mae_eidx_top3": "", "test_mae_eidx_top5": "",
                    "test_mse_epos_top1": "", "test_mse_epos_top3": "", "test_mse_epos_top5": "",
                })
            
            epoch_logs.append(epoch_log)

    # =============== Final Summary ===============
    if rank == 0:
        final_log = epoch_logs[-1]
        if final_log.get('test_ADE') != "":
            print("\n" + "=" * 80)
            print(f" FINAL TEST SUMMARY - {model_name} ({args.games} games)")
            print("=" * 80)
            
            # Get the last test_result for detailed metrics
            if test_result is not None:
                # Trajectory
                print(f" Trajectory Prediction:")
                print(f"  Short-term (3s): ADE={test_result['ADE_short']:.4f}, FDE={test_result['FDE_short']:.4f}")
                print(f"  Long-term (10s): ADE={test_result['ADE']:.4f}, FDE={test_result['FDE']:.4f}")
                
                # Event - Target player
                print(f" Target Player Accuracy:")
                print(f"  Top-1: {test_result['acc_tgt']:.4f}")
                print(f"  Top-3: {test_result['acc_tgt_top3']:.4f}")
                print(f"  Top-5: {test_result['acc_tgt_top5']:.4f}")
                
                # Event - Timing
                print(f"  Event Timing MAE (frames):")
                print(f"  Top-1: {test_result['mae_eidx']:.4f}")
                print(f"  Top-3: {test_result['mae_eidx_top3']:.4f}")
                print(f"  Top-5: {test_result['mae_eidx_top5']:.4f}")
                
                # Event - Position
                print(f" Event Position MSE (m²):")
                print(f"  Top-1: {test_result['mse_epos']:.4f}")
                print(f"  Top-3: {test_result['mse_epos_top3']:.4f}")
                print(f"  Top-5: {test_result['mse_epos_top5']:.4f}")
                
                # Disentanglement
                if 'ortho_loss' in test_result:
                    print(f"\n🔀 Disentanglement:")
                    print(f"  Orthogonality: {test_result.get('ortho_loss', 0.0):.4f}")
                    print(f"  Alignment:     {test_result.get('align_loss', 0.0):.4f}")
            
            print("=" * 80)
        
        csv_path = os.path.join(metrics_dir, f"metrics_{model_name}_{args.games}games.csv")
        with open(csv_path, "w", newline="") as f:
            csv_writer = csv.DictWriter(f, fieldnames=list(epoch_logs[0].keys()))
            csv_writer.writeheader()
            csv_writer.writerows(epoch_logs)
        print(f"✅ Saved metrics to {csv_path}")
    if rank == 0 and writer is not None:
        writer.flush()
        writer.close()


if __name__ == '__main__':
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

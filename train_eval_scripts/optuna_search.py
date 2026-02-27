from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import optuna
import torch
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_original, my_collate_fn_baseline
from models import CPTransformer
from utils import trackio_logging
from utils.tools import get_parameter_number

ACC_THRESHOLDS = (0.10, 0.15)


@dataclass
class CachedData:
    train_dataset: Dataset_original
    val_dataset: Dataset_original
    test_dataset: Dataset_original


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        # data
        dataset=args.dataset,
        data=args.data,
        root_path=args.root_path,
        num_workers=args.num_workers,
        dataloader_timeout=args.dataloader_timeout,
        weighted_sampling=args.weighted_sampling,
        weighted_loss=args.weighted_loss,
        seq_len=args.seq_len,
        charge_discharge_length=args.charge_discharge_length,
        train_epochs=args.train_epochs,
        patience=args.patience,
        # model/data defaults
        early_cycle_threshold=args.early_cycle_threshold,
        output_num=1,
        activation="relu",
        alpha1=0.15,
        alpha2=0.1,
        # fixed run defaults
        task_name="long_term_forecast",
        is_training=1,
        model="CPTransformer",
        model_id="optuna",
        model_comment="none",
        # placeholders overwritten per trial
        learning_rate=1e-4,
        wd=0.0,
        batch_size=32,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=2,
        d_ff=128,
        dropout=0.1,
        lradj="constant",
        pct_start=0.2,
        accumulation_steps=1,
        factor=1,
    )


def preview_files(files: list[str], max_items: int = 6) -> str:
    if not files:
        return "[]"
    if len(files) <= max_items:
        return f"[{', '.join(files)}]"
    head = ", ".join(files[: max_items // 2])
    tail = ", ".join(files[-max_items // 2 :])
    return f"[{head}, ..., {tail}]"


def describe_dataset(ds: Dataset_original, name: str) -> None:
    num_files = len(ds.files) if hasattr(ds, "files") else 0
    num_samples = len(ds)
    print(
        f"[data] {name}: files={num_files} samples={num_samples} "
        f"seq_len={ds.seq_len} early_cycle_threshold={ds.early_cycle_threshold} "
        f"charge_discharge_length={ds.charge_discharge_len}"
    )
    if num_files:
        print(f"[data] {name} file preview: {preview_files(ds.files)}")


def build_trial_args(base: SimpleNamespace, trial: optuna.Trial, seed: int) -> SimpleNamespace:
    cfg = SimpleNamespace(**vars(base))
    cfg.seed = seed
    cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    cfg.wd = trial.suggest_float("wd", 0.0, 1e-3)
    cfg.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    model_head_pairs = [
        (16, 2),
        (16, 4),
        (16, 8),
        (16, 16),
        (32, 2),
        (32, 4),
        (32, 8),
        (32, 16),
        (32, 32),
        (64, 2),
        (64, 4),
        (64, 8),
        (64, 16),
        (64, 32),
        (128, 2),
        (128, 4),
        (128, 8),
        (128, 16),
        (128, 32),
    ]
    cfg.d_model, cfg.n_heads = trial.suggest_categorical("model_heads", model_head_pairs)
    cfg.e_layers = trial.suggest_categorical("e_layers", [1, 2, 3, 4])
    cfg.d_layers = trial.suggest_categorical(
        "d_layers", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    cfg.d_ff = trial.suggest_categorical("d_ff", [32, 64, 128, 256])
    cfg.dropout = trial.suggest_float("dropout", 0.0, 0.3)
    cfg.lradj = trial.suggest_categorical("lradj", ["constant", "COS", "onecycle"])
    cfg.pct_start = trial.suggest_float("pct_start", 0.05, 0.3)
    cfg.accumulation_steps = trial.suggest_categorical("accumulation_steps", [1, 2, 4, 8])
    cfg.factor = trial.suggest_categorical("factor", [1, 2, 3, 5])
    cfg.model_comment = f"trial-{trial.number}"
    return cfg


def load_data(args: SimpleNamespace) -> CachedData:
    print(
        "[data] Loading datasets "
        f"dataset={args.dataset} root_path={args.root_path} "
        f"seq_len={args.seq_len} early_cycle_threshold={args.early_cycle_threshold} "
        f"charge_discharge_length={args.charge_discharge_length}"
    )
    print("[data] Loading train split...")
    train_dataset = Dataset_original(args=args, flag="train", label_scaler=None)
    describe_dataset(train_dataset, "train")
    label_scaler = train_dataset.return_label_scaler()
    life_class_scaler = train_dataset.return_life_class_scaler()
    print("[data] Loading val split...")
    val_dataset = Dataset_original(
        args=args, flag="val", label_scaler=label_scaler, life_class_scaler=life_class_scaler
    )
    describe_dataset(val_dataset, "val")
    print("[data] Loading test split...")
    test_dataset = Dataset_original(
        args=args, flag="test", label_scaler=label_scaler, life_class_scaler=life_class_scaler
    )
    describe_dataset(test_dataset, "test")
    return CachedData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def make_loader(
    dataset: Dataset_original,
    batch_size: int,
    num_workers: int,
    timeout: int,
    train: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
        collate_fn=my_collate_fn_baseline,
        generator=generator,
        worker_init_fn=seed_worker,
        timeout=timeout,
    )


def evaluate_loader(
    model: nn.Module,
    dataset: Dataset_original,
    loader: DataLoader,
    device: torch.device,
    alpha1: float,
) -> tuple[float, float, float, float]:
    preds = []
    refs = []
    std = np.sqrt(dataset.label_scaler.var_[-1])
    mean_value = dataset.label_scaler.mean_[-1]
    model.eval()
    with torch.no_grad():
        for cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights, seen_unseen_ids in loader:
            cycle_curve_data = cycle_curve_data.float().to(device)
            curve_attn_mask = curve_attn_mask.float().to(device)
            labels = labels.float().to(device)
            outputs = model(cycle_curve_data, curve_attn_mask)
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value
            preds.extend(transformed_preds.detach().cpu().numpy().reshape(-1).tolist())
            refs.extend(transformed_labels.detach().cpu().numpy().reshape(-1).tolist())
    model.train()
    rmse = root_mean_squared_error(refs, preds)
    mape = mean_absolute_percentage_error(refs, preds)
    acc10, acc15 = accuracy_at_pct(refs, preds)
    return rmse, mape, acc10, acc15


def accuracy_at_pct(refs: list[float], preds: list[float]) -> tuple[float, float]:
    if not refs:
        return 0.0, 0.0
    refs_arr = np.asarray(refs, dtype=np.float64)
    preds_arr = np.asarray(preds, dtype=np.float64)
    denom = np.maximum(np.abs(refs_arr), 1e-8)
    ape = np.abs(preds_arr - refs_arr) / denom
    acc10 = float(np.mean(ape <= ACC_THRESHOLDS[0]))
    acc15 = float(np.mean(ape <= ACC_THRESHOLDS[1]))
    return acc10, acc15


def train_one_trial(
    trial: optuna.Trial,
    args: SimpleNamespace,
    cached: CachedData,
    device: torch.device,
    trackio_project: str,
    trial_timeout_sec: int,
) -> tuple[float, int, float]:
    trial_start = time.time()
    train_loader = make_loader(
        cached.train_dataset,
        args.batch_size,
        args.num_workers,
        args.dataloader_timeout,
        train=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        cached.val_dataset,
        args.batch_size,
        args.num_workers,
        args.dataloader_timeout,
        train=False,
        seed=args.seed,
    )
    test_loader = make_loader(
        cached.test_dataset,
        args.batch_size,
        args.num_workers,
        args.dataloader_timeout,
        train=False,
        seed=args.seed,
    )

    model = CPTransformer.Model(args).float().to(device)
    para_res = get_parameter_number(model)
    model_size = int(para_res["Total"])

    model_optim = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.wd
    )
    if args.lradj == "COS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=max(args.train_epochs, 1), eta_min=1e-8
        )
    elif args.lradj == "onecycle":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=max(len(train_loader), 1),
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )
    else:
        scheduler = None

    criterion = nn.MSELoss(reduction="none")
    run_name = f"{args.model_comment}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    run_config = dict(vars(args))
    run_config["model_total_params"] = para_res["Total"]
    run_config["model_trainable_params"] = para_res["Trainable"]
    run_config["model_trainable_percent"] = para_res["Percent"]

    print(
        f"[trial {trial.number}] "
        f"seed={args.seed} device={device.type} "
        f"batch_size={args.batch_size} lr={args.learning_rate:.3e} wd={args.wd:.3e} "
        f"d_model={args.d_model} n_heads={args.n_heads} e_layers={args.e_layers} d_layers={args.d_layers} "
        f"d_ff={args.d_ff} dropout={args.dropout} lradj={args.lradj} "
        f"accumulation_steps={args.accumulation_steps}"
    )
    print(
        f"[trial {trial.number}] "
        f"train_samples={len(cached.train_dataset)} val_samples={len(cached.val_dataset)} "
        f"test_samples={len(cached.test_dataset)} steps_per_epoch={len(train_loader)}"
    )

    trackio_logging.init(project=trackio_project, config=run_config, name=run_name)
    best_val_mape = float("inf")
    best_test_mape = float("inf")
    best_time_sec = 0.0
    epochs_since_improve = 0
    try:
        global_step = 0
        for epoch in range(args.train_epochs):
            model.train()
            total_loss = 0.0
            std = np.sqrt(cached.train_dataset.label_scaler.var_[-1])
            mean_value = cached.train_dataset.label_scaler.mean_[-1]
            total_preds = []
            total_refs = []
            model_optim.zero_grad(set_to_none=True)
            for i, (cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights, seen_unseen_ids) in enumerate(train_loader):
                cycle_curve_data = cycle_curve_data.float().to(device)
                curve_attn_mask = curve_attn_mask.float().to(device)
                labels = labels.float().to(device)
                weights = weights.float().to(device)

                outputs = model(cycle_curve_data, curve_attn_mask)
                loss = criterion(outputs, labels)
                loss = torch.mean(loss * weights) / args.accumulation_steps
                loss.backward()

                if (i + 1) % args.accumulation_steps == 0:
                    model_optim.step()
                    model_optim.zero_grad(set_to_none=True)
                    if args.lradj == "onecycle" and scheduler is not None:
                        scheduler.step()
                total_loss += loss.detach().item() * args.accumulation_steps

                transformed_preds = outputs * std + mean_value
                transformed_labels = labels * std + mean_value
                total_preds.extend(transformed_preds.detach().cpu().numpy().reshape(-1).tolist())
                total_refs.extend(transformed_labels.detach().cpu().numpy().reshape(-1).tolist())
                global_step += 1

            if len(train_loader) % args.accumulation_steps != 0:
                model_optim.step()
                model_optim.zero_grad(set_to_none=True)
                if args.lradj == "onecycle" and scheduler is not None:
                    scheduler.step()

            if args.lradj == "COS" and scheduler is not None:
                scheduler.step()

            train_mape = mean_absolute_percentage_error(total_refs, total_preds)
            train_acc10, train_acc15 = accuracy_at_pct(total_refs, total_preds)
            train_loss = total_loss / max(len(train_loader), 1)
            val_rmse, val_mape, val_acc10, val_acc15 = evaluate_loader(
                model, cached.val_dataset, val_loader, device, args.alpha1
            )
            test_rmse, test_mape, test_acc10, test_acc15 = evaluate_loader(
                model, cached.test_dataset, test_loader, device, args.alpha1
            )
            if val_mape < best_val_mape:
                best_val_mape = val_mape
                best_test_mape = test_mape
                best_time_sec = time.time() - trial_start
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            print(
                f"[trial {trial.number}] epoch={epoch + 1}/{args.train_epochs} "
                f"train_loss={train_loss:.6f} train_mape={train_mape:.6f} "
                f"train_acc10={train_acc10:.4f} train_acc15={train_acc15:.4f} "
                f"val_rmse={val_rmse:.6f} val_mape={val_mape:.6f} "
                f"val_acc10={val_acc10:.4f} val_acc15={val_acc15:.4f} "
                f"test_rmse={test_rmse:.6f} test_mape={test_mape:.6f} "
                f"test_acc10={test_acc10:.4f} test_acc15={test_acc15:.4f}"
            )

            trackio_logging.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_mape": train_mape,
                    "train_acc10": train_acc10,
                    "train_acc15": train_acc15,
                    "vali_RMSE": val_rmse,
                    "vali_MAPE": val_mape,
                    "vali_acc10": val_acc10,
                    "vali_acc15": val_acc15,
                    "test_RMSE": test_rmse,
                    "test_MAPE": test_mape,
                    "test_acc10": test_acc10,
                    "test_acc15": test_acc15,
                },
                step=global_step,
            )

            if args.patience > 0 and epochs_since_improve >= args.patience:
                print(
                    f"[trial {trial.number}] early stop: no val_mape improvement for "
                    f"{args.patience} epochs"
                )
                break

            if trial_timeout_sec > 0 and (time.time() - trial_start) > trial_timeout_sec:
                raise RuntimeError(f"trial timed out after {trial_timeout_sec}s")

        train_time_sec = time.time() - trial_start
        trackio_logging.log(
            {
                "best_vali_MAPE": best_val_mape,
                "best_test_MAPE": best_test_mape,
                "train_time_sec": train_time_sec,
                "best_time_sec": best_time_sec,
                "model_total_params": model_size,
            }
        )
        return best_val_mape, model_size, best_time_sec
    finally:
        trackio_logging.finish()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="In-process Optuna search for CPTransformer on NAion")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_cptransformer_naion.db")
    parser.add_argument("--study-name", type=str, default="cptransformer_naion")
    parser.add_argument("--data", type=str, default="Dataset_original")
    parser.add_argument("--root-path", type=str, default="./dataset")
    parser.add_argument("--trackio-project", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataloader-timeout", type=int, default=300)
    parser.add_argument("--dataset", type=str, default="NAion")
    parser.add_argument("--early-cycle-threshold", type=int, default=100)
    parser.add_argument("--weighted_loss", action="store_true", default=False)
    parser.add_argument("--weighted_sampling", action="store_true", default=False)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--charge-discharge-length", type=int, default=100)
    parser.add_argument("--trial-timeout-sec", type=int, default=7200)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(1, 2**31 - 1)
    set_seed(seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            sampler=sampler,
            directions=["minimize", "minimize", "minimize"],
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc}\nUse a new --study-name for this objective setup or a new --storage database."
        ) from exc

    project = args.trackio_project or args.study_name
    print(f'Trackio: trackio show --project "{project}"')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_args = build_base_args(args)
    cached = load_data(base_args)

    def objective(trial: optuna.Trial) -> tuple[float, int, float]:
        trial_seed = seed + trial.number * 9973
        set_seed(trial_seed)
        trial_args = build_trial_args(base_args, trial, seed=trial_seed)
        if trial_args.d_model % trial_args.n_heads != 0:
            msg = f"invalid config: d_model={trial_args.d_model} not divisible by n_heads={trial_args.n_heads}"
            trial.set_user_attr("failed", msg)
            raise optuna.TrialPruned(msg)
        try:
            values = train_one_trial(
                trial=trial,
                args=trial_args,
                cached=cached,
                device=device,
                trackio_project=project,
                trial_timeout_sec=args.trial_timeout_sec,
            )
        except Exception as exc:
            msg = f"trial failed: {exc!r}"
            trial.set_user_attr("failed", msg)
            print(f"[trial {trial.number}] {msg}")
            raise optuna.TrialPruned(msg) from exc
        trial.set_user_attr("model_size", values[1])
        trial.set_user_attr("train_time_sec", values[2])
        trial.set_user_attr("trial_seed", trial_seed)
        return values

    try:
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True, catch=())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress is saved in Optuna storage; rerun the same command to resume.")
        return 130

    print("Pareto-optimal trials:")
    for t in study.best_trials:
        print(
            f"  trial={t.number} val_mape={t.values[0]:.6f} "
            f"model_size={int(t.values[1])} train_time_sec={t.values[2]:.2f} params={t.params}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

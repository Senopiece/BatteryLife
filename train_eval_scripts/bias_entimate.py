"""Estimate a constant-output baseline that mirrors Optuna trial logic.

This script uses the same Dataset_original sampling/scaling pipeline as
train_eval_scripts/optuna_search.py, but replaces CPTransformer with a
single constant prediction value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
from sklearn.metrics import root_mean_squared_error

# Allow running this script directly from train_eval_scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_loader import Dataset_original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a constant-output (bias-only) baseline aligned with Optuna."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NAion",
        help="Dataset key, e.g. NAion, NAion42, HUST, MATR, ZN-coin.",
    )
    parser.add_argument(
        "--root_path",
        type=Path,
        default=Path("./dataset"),
        help="Dataset root used by Dataset_original.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="Observed-cycle window size used in sample construction.",
    )
    parser.add_argument(
        "--charge_discharge_length",
        type=int,
        default=100,
        help="Resampled charge/discharge curve length used by the loader.",
    )
    parser.add_argument(
        "--early_cycle_threshold",
        type=int,
        default=100,
        help="Maximum observed cycle index used to create samples.",
    )
    parser.add_argument("--weighted_loss", action="store_true", default=False)
    parser.add_argument(
        "--fit_on",
        type=str,
        choices=["train", "train_val", "all"],
        default="train",
        help="Which split(s) to use to fit the constant (Optuna uses train).",
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        default=0.15,
        help="Relative error threshold for alpha1-accuracy.",
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        default=0.10,
        help="Relative error threshold for alpha2-accuracy.",
    )
    return parser.parse_args()


def build_loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        root_path=str(args.root_path),
        seq_len=args.seq_len,
        charge_discharge_length=args.charge_discharge_length,
        dataset=args.dataset,
        early_cycle_threshold=args.early_cycle_threshold,
        weighted_loss=args.weighted_loss,
    )


def load_datasets(args: argparse.Namespace) -> Dict[str, Dataset_original]:
    loader_args = build_loader_args(args)

    train_data = Dataset_original(
        args=loader_args,
        flag="train",
    )
    label_scaler = train_data.return_label_scaler()
    life_class_scaler = train_data.return_life_class_scaler()

    val_data = Dataset_original(
        args=loader_args,
        flag="val",
        label_scaler=label_scaler,
        life_class_scaler=life_class_scaler,
    )
    test_data = Dataset_original(
        args=loader_args,
        flag="test",
        label_scaler=label_scaler,
        life_class_scaler=life_class_scaler,
    )

    return {"train": train_data, "val": val_data, "test": test_data}


def split_std_labels_and_weights(dataset: Dataset_original) -> tuple[np.ndarray, np.ndarray]:
    labels_std = np.asarray(dataset.total_labels, dtype=np.float64).reshape(-1)
    weights = np.asarray(dataset.weights, dtype=np.float64).reshape(-1)
    if labels_std.size == 0:
        raise RuntimeError("Encountered empty split dataset.")
    if weights.size != labels_std.size:
        raise RuntimeError("weights and labels have different lengths.")
    return labels_std, weights


def fit_constant_std(
    split_data: Dict[str, tuple[np.ndarray, np.ndarray]], fit_on: str
) -> float:
    if fit_on == "train":
        y_std, w = split_data["train"]
    elif fit_on == "train_val":
        y_std = np.concatenate([split_data["train"][0], split_data["val"][0]])
        w = np.concatenate([split_data["train"][1], split_data["val"][1]])
    else:
        y_std = np.concatenate(
            [split_data["train"][0], split_data["val"][0], split_data["test"][0]]
        )
        w = np.concatenate(
            [split_data["train"][1], split_data["val"][1], split_data["test"][1]]
        )
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        raise RuntimeError("Sum of weights is not positive.")
    return float(np.sum(w * y_std) / w_sum)


def calc_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, alpha1: float, alpha2: float
) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)

    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))

    nonzero = y_true != 0
    if np.any(nonzero):
        rel_err = abs_err[nonzero] / np.abs(y_true[nonzero])
        mape = float(np.mean(rel_err))
        alpha1_acc = float(np.mean(rel_err <= alpha1) * 100.0)
        alpha2_acc = float(np.mean(rel_err <= alpha2) * 100.0)
    else:
        mape = float("nan")
        alpha1_acc = float("nan")
        alpha2_acc = float("nan")

    return {
        "count": int(len(y_true)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "alpha1_acc": alpha1_acc,
        "alpha2_acc": alpha2_acc,
    }


def main() -> None:
    args = parse_args()
    datasets = load_datasets(args)
    split_data = {
        split: split_std_labels_and_weights(ds) for split, ds in datasets.items()
    }
    constant_std = fit_constant_std(split_data, args.fit_on)
    train_scaler = datasets["train"].label_scaler
    std = float(np.sqrt(train_scaler.var_[-1]))
    mean_value = float(train_scaler.mean_[-1])
    constant_raw = constant_std * std + mean_value

    print(f"Dataset: {args.dataset}")
    print(f"seq_len: {args.seq_len}")
    print(f"charge_discharge_length: {args.charge_discharge_length}")
    print(f"early_cycle_threshold: {args.early_cycle_threshold}")
    print(f"weighted_loss: {args.weighted_loss}")
    print("Target: EOL (same label as Optuna training/evaluation).")
    print(f"Fitted constant in standardized space ({args.fit_on}): {constant_std:.6f}")
    print(f"Fitted constant in raw cycle-life space ({args.fit_on}): {constant_raw:.6f}")
    print(f"alpha1: {args.alpha1}, alpha2: {args.alpha2}")

    for split_name in ("train", "val", "test"):
        y_true_std, _ = split_data[split_name]
        y_true_raw = y_true_std * std + mean_value
        y_pred_raw = np.full_like(y_true_raw, fill_value=constant_raw)
        rmse_optuna_style = root_mean_squared_error(y_true_raw, y_pred_raw)
        metrics = calc_metrics(y_true_raw, y_pred_raw, args.alpha1, args.alpha2)

        print(f"\n[{split_name}]")
        print(f"count: {metrics['count']}")
        print(f"rmse_optuna_style: {rmse_optuna_style:.6f}")
        print(f"mse: {metrics['mse']:.6f}")
        print(f"rmse: {metrics['rmse']:.6f}")
        print(f"mae: {metrics['mae']:.6f}")
        print(f"mape: {metrics['mape']:.6f}")
        print(f"alpha1-accuracy ({args.alpha1}): {metrics['alpha1_acc']:.2f}%")
        print(f"alpha2-accuracy ({args.alpha2}): {metrics['alpha2_acc']:.2f}%")


if __name__ == "__main__":
    main()

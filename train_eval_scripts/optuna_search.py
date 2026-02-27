#!./.venv/bin/python3
from __future__ import annotations

import argparse
import random
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List

import optuna


BEST_RE = re.compile(r"Val MAPE: ([0-9.]+)")
MODEL_SIZE_RE = re.compile(r"(?:'Total'|\"Total\"|Total)\s*:\s*([0-9]+)")


@dataclass
class TrialResult:
    val_mape: float
    model_size: int
    stdout: str


def run_training(cmd: List[str]) -> TrialResult:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(f"training failed (code={proc.returncode}). Output:\n{out}")
    match = None
    for line in reversed(out.splitlines()):
        match = BEST_RE.search(line)
        if match:
            break
    if not match:
        raise RuntimeError(f"could not parse Val MAPE from output.\nOutput tail:\n{out[-4000:]}")
    size_match = None
    for line in out.splitlines():
        size_match = MODEL_SIZE_RE.search(line)
        if size_match:
            break
    if not size_match:
        raise RuntimeError(f"could not parse model size from output.\nOutput tail:\n{out[-4000:]}")
    return TrialResult(
        val_mape=float(match.group(1)),
        model_size=int(size_match.group(1)),
        stdout=out,
    )


def build_command(args: argparse.Namespace, trial: optuna.Trial) -> List[str]:
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    wd = trial.suggest_float("wd", 0.0, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16, 24, 32])
    e_layers = trial.suggest_categorical("e_layers", [1, 2, 3, 4])
    d_layers = trial.suggest_categorical("d_layers", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    d_ff = trial.suggest_categorical("d_ff", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lradj = trial.suggest_categorical("lradj", ["constant", "COS"])
    pct_start = trial.suggest_float("pct_start", 0.05, 0.3)
    seq_len = trial.suggest_categorical("seq_len", [5, 10, 20])

    cmd = [
        args.python,
        "run_main.py",
        "--model",
        "CPTransformer",
        "--dataset",
        "NAion",
        "--data",
        args.data,
        "--is_training",
        "1",
        "--itr",
        "1",
        "--model_id",
        "optuna",
        "--model_comment",
        f"optuna-trial-{trial.number}",
        "--seed",
        str(args.seed),
        "--trackio_project",
        args.trackio_project or args.study_name,
        "--root_path",
        args.root_path,
        "--learning_rate",
        str(lr),
        "--wd",
        str(wd),
        "--batch_size",
        str(batch_size),
        "--d_model",
        str(d_model),
        "--n_heads",
        str(n_heads),
        "--e_layers",
        str(e_layers),
        "--d_layers",
        str(d_layers),
        "--d_ff",
        str(d_ff),
        "--dropout",
        str(dropout),
        "--lradj",
        str(lradj),
        "--pct_start",
        str(pct_start),
        "--seq_len",
        str(seq_len),
    ]

    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna search for CPTransformer on NAion")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_cptransformer_naion.db")
    parser.add_argument("--study-name", type=str, default="cptransformer_naion")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--data", type=str, default="Dataset_original")
    parser.add_argument("--root-path", type=str, default="./dataset")
    parser.add_argument("--trackio-project", type=str, default="")
    parser.add_argument("--extra-args", type=str, default="")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(1, 2**31 - 1)
    args.seed = seed
    sampler = optuna.samplers.TPESampler(seed=seed)
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            sampler=sampler,
            directions=["minimize", "minimize"],
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc}\nUse a new --study-name for multi-objective search or a new --storage database."
        ) from exc
    project = args.trackio_project or args.study_name
    print(f'Trackio: trackio show --project "{project}"')

    def objective(trial: optuna.Trial) -> tuple[float, int]:
        cmd = build_command(args, trial)
        result = run_training(cmd)
        trial.set_user_attr("command", " ".join(shlex.quote(c) for c in cmd))
        trial.set_user_attr("model_size", result.model_size)
        return result.val_mape, result.model_size

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True, catch=(RuntimeError,))

    print("Pareto-optimal trials:")
    for t in study.best_trials:
        print(
            f"  trial={t.number} val_mape={t.values[0]:.6f} model_size={int(t.values[1])} "
            f"params={t.params}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

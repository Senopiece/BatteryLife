#!./.venv/bin/python3
from __future__ import annotations

import argparse
import collections
import random
import re
import shlex
import subprocess
import sys
import time
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


def run_training(cmd: List[str], trial_timeout_sec: int, idle_timeout_sec: int) -> TrialResult:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("failed to capture subprocess stdout")

    start_time = time.time()
    last_output_time = start_time
    tail = collections.deque(maxlen=2000)
    best_match = None
    size_match = None

    try:
        while True:
            line = proc.stdout.readline()
            now = time.time()
            if line:
                tail.append(line)
                last_output_time = now
                m = BEST_RE.search(line)
                if m:
                    best_match = m
                s = MODEL_SIZE_RE.search(line)
                if s:
                    size_match = s
            elif proc.poll() is not None:
                break

            elapsed = now - start_time
            idle = now - last_output_time
            if trial_timeout_sec > 0 and elapsed > trial_timeout_sec:
                proc.kill()
                raise RuntimeError(
                    f"trial timed out after {trial_timeout_sec}s. Output tail:\n{''.join(tail)}"
                )
            if idle_timeout_sec > 0 and idle > idle_timeout_sec:
                proc.kill()
                raise RuntimeError(
                    f"trial produced no output for {idle_timeout_sec}s. Output tail:\n{''.join(tail)}"
                )

        remaining = proc.stdout.read()
        if remaining:
            tail.append(remaining)
            m = BEST_RE.search(remaining)
            if m:
                best_match = m
            s = MODEL_SIZE_RE.search(remaining)
            if s:
                size_match = s
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        raise
    finally:
        proc.stdout.close()

    if proc.returncode != 0:
        raise RuntimeError(
            f"training failed (code={proc.returncode}). Output tail:\n{''.join(tail)}"
        )
    if not best_match:
        raise RuntimeError(
            f"could not parse Val MAPE from output. Output tail:\n{''.join(tail)}"
        )
    if not size_match:
        raise RuntimeError(
            f"could not parse model size from output. Output tail:\n{''.join(tail)}"
        )
    return TrialResult(
        val_mape=float(best_match.group(1)),
        model_size=int(size_match.group(1)),
        stdout="".join(tail),
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
    pred_len = trial.suggest_categorical("pred_len", [3, 5, 10, 20])
    accumulation_steps = trial.suggest_categorical("accumulation_steps", [1, 2, 4, 8])
    factor = trial.suggest_categorical("factor", [1, 2, 3, 5])
    charge_discharge_length = trial.suggest_categorical("charge_discharge_length", [80, 100, 120, 150])

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
        f"trial-{trial.number}",
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
        "--pred_len",
        str(pred_len),
        "--accumulation_steps",
        str(accumulation_steps),
        "--factor",
        str(factor),
        "--charge_discharge_length",
        str(charge_discharge_length),
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
    parser.add_argument("--trial-timeout-sec", type=int, default=7200)
    parser.add_argument("--idle-timeout-sec", type=int, default=1800)
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
        result = run_training(
            cmd, trial_timeout_sec=args.trial_timeout_sec, idle_timeout_sec=args.idle_timeout_sec
        )
        trial.set_user_attr("command", " ".join(shlex.quote(c) for c in cmd))
        trial.set_user_attr("model_size", result.model_size)
        return result.val_mape, result.model_size

    try:
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True, catch=(RuntimeError,))
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress is saved in Optuna storage; rerun the same command to resume.")
        return 130

    print("Pareto-optimal trials:")
    for t in study.best_trials:
        print(
            f"  trial={t.number} val_mape={t.values[0]:.6f} model_size={int(t.values[1])} "
            f"params={t.params}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

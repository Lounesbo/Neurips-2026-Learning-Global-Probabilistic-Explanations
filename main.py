# -----------------------------------------------------------------------------
#
# Experimental series of the paper
#
# Every series is parameterized by (epsilon, delta) only, as in the theory; the
# dimensions (d, k, c) are read off the task, kappa is the single calibrated
# constant, and the machine cap is given in minutes per run (--minutes).
#
#   1 eps_sweep     gap err(K) - opt against epsilon, with the line y = epsilon
#   2 delta_sweep   failure frequency against delta, with the line y = delta
#   3 d_sweep       queries and IKM calls against d
#   4 k_sweep       queries and IKM calls against k
#   5 c_sweep       queries and IKM calls against c
#   6 trap          SE against GE on the Synergy Trap, as k grows
#   7 datasets      SE against GE on OpenML, as k grows; the table of the paper
#                   is the k = 5 slice of the same runs (displayer.py --table 5)
#
# Usage:  python main.py --series 6 --gpus 0 1        (both GPUs at once, calibration included)
#         python main.py --series 1 2 --calibrate      (by hand: once, measures throughputs,
#         python main.py --series 1 2 --device cuda:0 --shard 0/2      then one terminal
#         python main.py --series 1 2 --device cuda:1 --shard 1/2      per GPU)
#         python main.py --series 7 --datasets 40536 --runs 1 --minutes 10   (quick test)
#         python main.py --inspect        (dataset statistics after binarization)
#
# -----------------------------------------------------------------------------

import argparse
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from blackbox import Classifier, NoisyParity, Parity, SynergyTrap
from dataset import Dataset
from distribution import ProductDistribution
from runner import Runner, Task
from utils import Console


# -- Configurations ----------------------------------------------------------------

@dataclass
class BaseConfig:
    prefix: str = "base"
    sweep_key: str = "none"
    seed: int = 48
    epsilon: float = 0.35
    delta: float = 0.1
    k: int = 3
    # Calibrated once on the parity benchmark (kappa = 0.003 is the smallest
    # value with no failure over five seeds; 0.01 leaves a factor 3 of margin).
    kappa: float = 0.01
    n_runs: int = 10
    ge_samples: int = 5000
    eval_samples: int = 1_000_000
    explainers: List[str] = field(default_factory=lambda: ["SE", "GE"])


EPSILONS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]   # err <= 1/2: eps >= 1/2 is vacuous
DELTAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
DIMENSIONS = [8, 16, 32, 64, 128]
SIZES = [1, 2, 3, 4, 5]
ALPHABETS = [2, 4, 6]                              # even, so that parities are balanced
# Interpretable features, d <= 20, so that opt is exact (brute force). The
# paper's results were produced in three campaigns (see README), and the seed
# of a run depends on the position of its point in the series: reproduce them
# with the same --datasets lists.
# Pima 8, Breast Cancer (WI) 8, Tic-Tac-Toe 9, Heart Disease 11, Adult 11,
# COMPAS 13, Bank Marketing 16, Mushroom 20, Credit-G 20.
DATASETS = [37, 15, 50, 43672, 179, 42192, 1461, 24, 31]


@dataclass
class EpsSweep(BaseConfig):
    prefix: str = "eps_sweep"
    sweep_key: str = "epsilon"
    d: int = 16
    eta: float = 0.05


@dataclass
class DeltaSweep(EpsSweep):
    prefix: str = "delta_sweep"
    sweep_key: str = "delta"
    n_runs: int = 100


@dataclass
class DSweep(BaseConfig):
    prefix: str = "d_sweep"
    sweep_key: str = "d"


@dataclass
class KSweep(BaseConfig):
    prefix: str = "k_sweep"
    sweep_key: str = "k"
    d: int = 16


@dataclass
class CSweep(BaseConfig):
    prefix: str = "c_sweep"
    sweep_key: str = "c"
    d: int = 16
    k: int = 2


@dataclass
class Trap(BaseConfig):
    prefix: str = "trap"
    sweep_key: str = "k"
    d: int = 16


@dataclass
class Datasets(BaseConfig):
    prefix: str = "datasets"
    sweep_key: str = "k"
    classifier: str = "Neural Network"
    dataset_ids: List[int] = field(default_factory=lambda: list(DATASETS))
    # Machine cap per dataset, in minutes, overriding --minutes for the large
    # ones, e.g. {1486: 60}. Every run of a point uses the same cap.
    minutes: Dict[int, float] = field(default_factory=dict)


def _dataset_overrides(config, openml_id: int, **extra) -> dict:
    overrides = dict(extra)
    if openml_id in config.minutes:
        overrides["minutes"] = config.minutes[openml_id]
    return overrides


# -- Tasks -----------------------------------------------------------------------------

def parity_task(d: int, c: int, size: int, device: str, seed: int = 0) -> Task:
    """Hidden parity on `size` features drawn at random among d; opt = 0 iff k >= size."""
    g = torch.Generator().manual_seed(seed + 7919 * d + size)
    support = sorted(torch.randperm(d, generator=g)[:size].tolist())
    return Task(name=f"parity_d{d}_c{c}_s{size}", f=Parity(support),
                D=ProductDistribution.uniform(d, c, device=device),
                opt=lambda k, s=size: 0.0 if k >= s else 0.5,
                info={"d": d, "c": c, "support": "|".join(map(str, support))})


def noisy_parity_task(config, device: str) -> Task:
    g = torch.Generator().manual_seed(config.seed)
    support = sorted(torch.randperm(config.d, generator=g)[:config.k].tolist())
    f = NoisyParity(support, eta=config.eta, d=config.d, seed=config.seed)
    # d = 16: |Omega| = 65536, so opt is computed exactly by brute force.
    return Task(name=f"noisy_parity_d{config.d}_eta{config.eta}", f=f,
                D=ProductDistribution.uniform(config.d, 2, device=device),
                info={"d": config.d, "c": 2, "eta": config.eta, "support": "|".join(map(str, support))})


def trap_task(d: int, size: int, device: str) -> Task:
    support = list(range(1, size + 1))
    return Task(name=f"trap_d{d}_s{size}", f=SynergyTrap(0, support),
                D=ProductDistribution.uniform(d, 2, device=device),
                info={"d": d, "c": 2, "support": "0|" + "|".join(map(str, support))})


def dataset_task(openml_id: int, config, device: str, console: Console) -> Task:
    ds = Dataset(id=openml_id, seed=config.seed, device=device, console=console).build()
    clf = Classifier(config.classifier, seed=config.seed)
    clf.train(ds.X_train, ds.Y_train, ds.X_test, ds.Y_test)
    info = ds.summary()
    info.update(classifier=config.classifier, train_accuracy=clf.train_accuracy,
                test_accuracy=clf.test_accuracy)
    return Task(name=ds.name, f=clf, D=ds.distribution, info=info)


# -- Series ------------------------------------------------------------------------------

def points_for(key: int, config, device: str, console: Console) -> list:
    if key == 1:
        task = noisy_parity_task(config, device)
        return [(e, task, {"epsilon": e}) for e in EPSILONS]
    if key == 2:
        task = noisy_parity_task(config, device)
        return [(dl, task, {"delta": dl}) for dl in DELTAS]
    if key == 3:
        return [(d, parity_task(d, 2, config.k, device), {}) for d in DIMENSIONS]
    if key == 4:
        return [(k, parity_task(config.d, 2, k, device), {"k": k}) for k in SIZES]
    if key == 5:
        return [(c, parity_task(config.d, c, config.k, device), {}) for c in ALPHABETS]
    if key == 6:
        return [(k, trap_task(config.d, k - 1, device), {"k": k}) for k in SIZES if k >= 2]
    if key == 7:
        points = []
        for i in config.dataset_ids:
            task = dataset_task(i, config, device, console)
            points += [(k, task, _dataset_overrides(config, i, k=k)) for k in SIZES]
        return points
    raise ValueError(key)


SERIES = {
    1: EpsSweep, 2: DeltaSweep, 3: DSweep, 4: KSweep, 5: CSweep,
    6: Trap, 7: Datasets,
}


def inspect(device: str, console: Console) -> None:
    import traceback
    import pandas
    import sklearn
    console.log("Versions", f"pandas {pandas.__version__}, scikit-learn {sklearn.__version__}")
    print()
    for i in DATASETS + [d for d in Dataset.OPENML if d not in DATASETS]:
        try:
            ds = Dataset(id=i, device=device).build()
            console.log(ds.summary())
            print()
        except Exception as error:  # a failed dataset should not stop the listing
            console.log(f"{Dataset.OPENML.get(i, i)}", f"failed: {error}", color="red")
            print("".join(traceback.format_exc().splitlines(keepends=True)[-6:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experimental series of the paper.")
    parser.add_argument("--series", nargs="*", type=int, default=[], help="Series numbers (see header).")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--minutes", type=float, default=20.0, help="Machine cap of one IKM run.")
    parser.add_argument("--shard", default="0/1", help="i/n: run the runs whose index is i mod n.")
    parser.add_argument("--inspect", action="store_true", help="Print dataset statistics and exit.")
    parser.add_argument("--runs", type=int, default=None, help="Override the number of runs per point.")
    parser.add_argument("--datasets", nargs="*", type=int, default=None,
                        help="OpenML ids overriding the dataset list of series 7.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Measure and store the throughput of every task of the series, then exit. "
                             "Run it once before starting several shards.")
    parser.add_argument("--tag", default=None, help="Campaign tag in the CSV names (set by --gpus). Reusing the tag of an "
                             "interrupted campaign resumes it.")
    parser.add_argument("--gpus", nargs="*", type=int, default=None,
                        help="Run the series on several GPUs at once, e.g. --gpus 0 1: calibrates first, "
                             "then starts one process per GPU, each on its own shard.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    console = Console(verbose=True)
    if args.inspect:
        inspect(args.device, console)
        return
    if args.gpus:
        launch(args, console)
        return

    runner = Runner(console, device=args.device, minutes=args.minutes, shard=args.shard, tag=args.tag)
    for key in args.series:
        config = configure(key, args)
        console.log(f"Series {key}: {config.prefix}", color="green")
        points = points_for(key, config, args.device, console)
        if args.calibrate:
            runner.calibrate(points)
        else:
            runner.run(config, points)


def configure(key: int, args):
    config = SERIES[key]()
    if args.runs is not None:
        config.n_runs = args.runs
    if args.datasets and hasattr(config, "dataset_ids"):
        config.dataset_ids = list(args.datasets)
    return config


def gpu_device(gpu: int) -> str:
    return f"cuda:{gpu}"


def launch(args, console: Console) -> None:
    """
    One process per GPU, all at the same time. The throughputs are measured
    first, in this process, so that every shard reads the same caps from
    exps/rates.json; each shard then writes its log to exps/logs/.
    """
    import subprocess
    import sys
    from pathlib import Path

    device = gpu_device(args.gpus[0])
    calibration = Runner(console, device=device, minutes=args.minutes)
    for key in args.series:
        calibration.calibrate(points_for(key, configure(key, args), device, console))

    logs = Path("exps") / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # An explicit --tag continues that campaign: each shard keeps its finished runs.
    stamp = args.tag or time.strftime("%Y%m%d-%H%M%S")
    series = "-".join(map(str, args.series))
    common = ["--series", *map(str, args.series), "--minutes", str(args.minutes), "--tag", stamp]
    if args.runs is not None:
        common += ["--runs", str(args.runs)]
    if args.datasets:
        common += ["--datasets", *map(str, args.datasets)]

    processes = []
    for i, gpu in enumerate(args.gpus):
        log = logs / f"series{series}_{stamp}_gpu{gpu}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *common,
                   "--device", gpu_device(gpu), "--shard", f"{i}/{len(args.gpus)}"]
        handle = open(log, "a")
        processes.append((gpu, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT), handle, log))
        console.log(f"GPU {gpu}: shard {i}/{len(args.gpus)}", f"log in {log}", color="cyan")

    for gpu, process, handle, log in processes:
        code = process.wait()
        handle.close()
        console.log(f"GPU {gpu}", "done" if code == 0 else f"failed (exit {code}), see {log}",
                    color="green" if code == 0 else "red")


if __name__ == "__main__":
    main()

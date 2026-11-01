# -----------------------------------------------------------------------------
#
# Runner
#
# One generic driver for every series. A series is a list of points, one per
# value of the swept parameter; at each point, a task (black box f, product
# distribution D, optional exact opt) is explained n_runs times by SE and GE,
# and each explanation is scored by the evaluator. Rows are appended to the CSV
# as they come, so that a long campaign can be stopped and inspected at any
# time; with --shard i/n, each process runs the runs whose index is i mod n,
# which spreads a series over the two GPUs.
#
# Seeds. The generator of a run depends on (series seed, point, run) only, so
# that every explainer and every point is reproducible on its own.
#
# Throughput. The cap of a run depends on the measured throughput, so that two
# shards measuring it separately would run the same point with different block
# sizes. Rates are therefore stored in <output_dir>/rates.json, keyed by task
# and device type, and read back by every shard: run --calibrate once before
# starting the shards.
#
# The CSV is rewritten after every run with the union of the columns seen so
# far, so that it is always complete and readable during a campaign. The write
# goes to a temporary file first, then replaces the CSV, so that a crash never
# leaves a truncated file; a failed write is retried, then reported, and never
# stops the campaign (the next run writes everything again).
#
# Resuming. If the CSV of this tag and shard already exists, its completed runs
# are kept and skipped: rerunning a campaign with the same --tag continues it.
# Since seeds depend on (point, run) only, the result is the same as that of an
# uninterrupted campaign.
#
# -----------------------------------------------------------------------------

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch

from blackbox import BlackBox
from budget import measure_throughput, query_cap
from distribution import ProductDistribution
from evaluator import Evaluator
from explainer import BruteForceExplainer, GreedyExplainer, SpectralExplainer
from oracle import MembershipOracle
from parameters import SEParameters
from utils import Console


@dataclass
class Task:
    name: str
    f: BlackBox
    D: ProductDistribution
    # Exact optimum by k, when it is known in closed form; otherwise it is
    # computed by brute force whenever |Omega| is small enough.
    opt: Optional[Callable[[int], float]] = None
    info: Dict[str, object] = field(default_factory=dict)


class Runner:
    def __init__(self, console: Console, device: str = "cpu", minutes: float = 20.0,
                 shard: str = "0/1", output_dir: str = "./exps", tag: Optional[str] = None):
        self.console = console
        self.device = device
        self.minutes = minutes
        self.shard, self.n_shards = (int(v) for v in shard.split("/"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Shared by all the shards of one campaign, so that their files can be
        # told apart from those of earlier campaigns in the same folder.
        self.tag = tag or time.strftime("%Y%m%d-%H%M%S")
        self._rates_path = self.output_dir / "rates.json"
        self._opts: Dict[tuple, Optional[float]] = {}

    # -- Budget and optimum ----------------------------------------------------

    def _load_rates(self) -> Dict[str, float]:
        if self._rates_path.exists():
            return json.loads(self._rates_path.read_text())
        return {}

    def rate(self, task: Task) -> float:
        """Queries per second of IKM on this task, measured once and shared by all shards."""
        key = f"{task.name}@{task.D.device.type}"
        rates = self._load_rates()
        if key not in rates:
            rates[key] = measure_throughput(task.f, task.D)
            rates.update({k: v for k, v in self._load_rates().items() if k != key})
            self._rates_path.write_text(json.dumps(rates, indent=2, sort_keys=True))
            self.console.log(f"Throughput on {task.name} (measured)", f"{rates[key]:.3e} queries/s")
        return rates[key]

    def calibrate(self, points: List[tuple]) -> None:
        """Measures and stores the throughput of every task of a series, nothing else."""
        seen = set()
        for _, task, _ in points:
            if task.name not in seen:
                seen.add(task.name)
                self.console.log(f"Throughput on {task.name}", f"{self.rate(task):.3e} queries/s")

    def opt(self, task: Task, evaluator: Evaluator, k: int) -> Optional[float]:
        key = (task.name, k)
        if key not in self._opts:
            if task.opt is not None:
                self._opts[key] = task.opt(k)
            elif evaluator.is_exact:
                bf = BruteForceExplainer(evaluator, k)
                bf.explain()
                self._opts[key] = bf.opt
                self.console.log(f"Brute force opt on {task.name} (k={k})", f"{bf.opt:.4f}")
            else:
                self._opts[key] = None
        return self._opts[key]

    # -- Driver ------------------------------------------------------------------

    def run(self, config, points: List[tuple]) -> Path:
        """
        config : dataclass of the series (see main.py).
        points : list of (value, task, overrides), where overrides set any of
                 epsilon, delta, k for this point.
        """
        path = self._output_path(config.prefix)
        rows, done = self._resume(path, config)
        for p_index, (value, task, overrides) in enumerate(points):
            pending = [run for run in range(config.n_runs)
                       if run % self.n_shards == self.shard and (p_index, task.name, run) not in done]
            if not pending:
                continue
            epsilon = overrides.get("epsilon", config.epsilon)
            delta = overrides.get("delta", config.delta)
            k = overrides.get("k", config.k)
            minutes = overrides.get("minutes", self.minutes)
            evaluator = Evaluator(task.f, task.D, n_samples=config.eval_samples, seed=config.seed)
            opt = self.opt(task, evaluator, k)
            rate = self.rate(task)
            params = SEParameters(epsilon=epsilon, delta=delta, k=k, d=task.D.d, c=task.D.c,
                                  kappa=config.kappa, max_queries=query_cap(rate, minutes))
            self.console.log(
                f"{config.prefix} | {config.sweep_key}={value} | {task.name}",
                f"n={params.ikm.n} b={params.ikm.b} capped={params.ikm.capped}", color="blue")

            for run in pending:
                for row in self._run_once(config, task, evaluator, params, opt, p_index, run):
                    row.update({"series": config.prefix, "sweep_key": config.sweep_key,
                                "value": value, "task": task.name, "point": p_index,
                                "run": run, "rate": rate, "minutes": minutes,
                                "max_queries": params.max_queries, **task.info})
                    rows.append(row)
                self._save(rows, path)
            if task.D.device.type == "cuda":
                torch.cuda.empty_cache()  # chunk sizes change with the point
        self.console.log("Saved", str(path), color="green")
        return path

    def _resume(self, path: Path, config) -> tuple:
        """Rows already in the CSV, and the (point, task, run) whose explainers are all there."""
        if not path.exists():
            return [], set()
        df = pd.read_csv(path)
        rows = df.to_dict("records")
        counts = df.groupby(["point", "task", "run"])["explainer"].nunique()
        done = {key for key, n in counts.items() if n >= len(config.explainers)}
        self.console.log("Resuming", f"{path} ({len(done)} runs done)", color="yellow")
        keep = [row for row in rows if (row["point"], row["task"], row["run"]) in done]
        return keep, done

    def _save(self, rows: List[dict], path: Path, attempts: int = 5) -> None:
        tmp = path.with_suffix(".tmp")
        for attempt in range(attempts):
            try:
                pd.DataFrame(rows).to_csv(tmp, index=False)
                os.replace(tmp, path)
                return
            except OSError as error:
                last = error
                time.sleep(2 * (attempt + 1))
        self.console.log("Save failed (retried at the next run)", str(last), color="red")

    def _run_once(self, config, task: Task, evaluator: Evaluator, params: SEParameters,
                  opt: Optional[float], p_index: int, run: int) -> List[dict]:
        seed = config.seed + 100_000 * p_index + run
        rows = []
        explainers = {
            "SE": lambda o: SpectralExplainer(o, task.D, params),
            "GE": lambda o: GreedyExplainer(o, task.D, params.k, n_samples=config.ge_samples),
        }
        for name in config.explainers:
            oracle = MembershipOracle(task.f)
            explainer = explainers[name](oracle)
            generator = torch.Generator(device=task.D.device).manual_seed(seed)
            K = explainer.explain(generator)
            err = evaluator.error(K)
            gap = None if opt is None else err - opt
            row = {
                "explainer": name, "seed": seed, "K": "|".join(task.D.names[i] for i in K),
                "size": len(K), "err": err, "opt": opt, "gap": gap,
                "success": None if gap is None else gap <= params.epsilon,
                "queries": explainer.n_queries, "walltime": explainer.walltime,
                "eval_exact": evaluator.is_exact,
            }
            row.update(params.as_dict())
            row.update(explainer.stats())
            rows.append(row)
            self.console.log(f"  run {run} {name}", f"err={err:.4f} K={K}")
        return rows

    def _output_path(self, prefix: str) -> Path:
        folder = self.output_dir / prefix
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{prefix}_{self.tag}_shard{self.shard}of{self.n_shards}.csv"

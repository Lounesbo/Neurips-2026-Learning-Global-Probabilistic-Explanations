import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackbox import Parity
from distribution import ProductDistribution
from main import BaseConfig
from runner import Runner, Task
from utils import Console


def test_rerun_with_same_tag_resumes(tmp_path):
    config = BaseConfig(prefix="toy", k=1, n_runs=3, eval_samples=10_000, ge_samples=500)
    D = ProductDistribution.uniform(6, 2)
    points = [(1, Task(name="parity", f=Parity([2]), D=D, opt=lambda k: 0.0), {})]
    quiet = Console(verbose=False)

    first = Runner(quiet, minutes=0.05, output_dir=tmp_path, tag="t")
    path = first.run(config, points)
    full = pd.read_csv(path)
    # simulate a crash after the first run
    full[full["run"] == 0].to_csv(path, index=False)

    second = Runner(quiet, minutes=0.05, output_dir=tmp_path, tag="t")
    resumed = pd.read_csv(second.run(config, points))
    assert len(resumed) == len(full) == 6
    assert (resumed["K"].fillna("").tolist() == full["K"].fillna("").tolist())
    assert (resumed["err"] - full["err"]).abs().max() < 1e-12

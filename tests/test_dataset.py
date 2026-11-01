import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import Dataset
from parameters import SEParameters


def frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.exponential(30_000, n),
        "sex": rng.choice(["Male", "Female"], n),
        "workclass": pd.Categorical(rng.choice(["Private", "Gov", "Self", "Other"], n, p=[.6, .2, .1, .1])),
        "parties": rng.choice(["none", "guarantor"], n),      # 'none' is a value
        "flag": np.ones(n),                                   # constant
        "broken": np.where(rng.random(n) < 0.1, np.nan, 1.0),  # incomplete
    })
    y = pd.Series(np.where(X["age"] > 40, ">50K", "<=50K"))
    return X, y


def test_binarization():
    X, y = frame()
    ds = Dataset(name="Toy").build_from_frame(X, y)
    assert ds.d == 5 and set(np.unique(ds.X)) <= {0, 1}
    assert any(name.startswith("parties=") for name in ds.names)
    assert ds.names[0].startswith("age>") and ds.names[1].startswith("income>")
    assert ds.names[2] in ("sex=Male", "sex=Female") and ds.names[3] == "workclass=Private"
    assert ds.n_dropped_incomplete == 1 and ds.n_dropped_constant == 1
    assert set(np.unique(ds.Y)) == {-1, 1}
    assert ds.distribution.c == 2 and ds.distribution.d == 5
    # median split: both values carry about half of the mass
    assert abs(ds.distribution.bases[0].mu[1] - 0.5) < 0.05


def test_cap_lowers_n_only():
    free = SEParameters(epsilon=0.3, delta=0.1, k=5, d=20, c=2, kappa=0.01)
    capped = SEParameters(epsilon=0.3, delta=0.1, k=5, d=20, c=2, kappa=0.01, max_queries=10 ** 9)
    assert not free.ikm.capped and capped.ikm.capped
    assert capped.ikm.b == free.ikm.b
    assert capped.ikm.queries <= 10 ** 9 < free.ikm.queries
    assert capped.ikm.kappa_effective < 0.01

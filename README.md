# Learning Global Probabilistic Explanations

**Frédéric Koriche** (CRIL, Univ. Artois, CNRS) · **Louenas Bounia** (LIPN, Université Sorbonne Paris Nord, CNRS)

*40th Conference on Neural Information Processing Systems (NeurIPS 2026)*

[**Paper (OpenReview)**](https://openreview.net/forum?id=c2Ny586AOf) · [Code](code/) · [Results](results/)

---

## Overview

Given a black-box classifier f and a distribution D over the inputs, a **global
probabilistic explanation** is a small subset K of features such that two
random inputs agreeing on K are likely to receive the same label. Its
**global relevance error** is

    err(K) = P_{x ~ D, z ~ D(· | z_K = x_K)} [ f(x) ≠ f(z) ]

and the goal is to find, under a size limit |K| ≤ k, an explanation of
minimal error.

For product distributions with full support over discrete feature spaces, the
paper shows that this error is entirely determined by the Fourier spectrum of f:

    err(K) = (1 − W(f_K)) / 2

where W(f_K) is the spectral energy of f on the coordinates supported on K.
Building on this identity, the **Spectral Explainer (SE)** combines:

1. **Spectral recovery**: an *Integer Kushilevitz–Mansour* (IKM) algorithm
   that finds the significant Fourier coefficients of f over general discrete
   domains, using membership queries and median-of-means estimates;
2. **Feature pruning**: keeping only the features with a large restricted
   influence;
3. **Subset selection**: an exhaustive search for the subset of maximal energy
   among the remaining features.

With probability at least 1 − δ, SE returns an explanation whose error is
within ε of the optimum. It is **fixed-parameter tractable** in k, the maximum
feature cardinality c and 1/ε, with a running time polynomial in d,
(ck/ε)^k and ln(1/δ).

Experiments on synthetic targets and nine OpenML datasets confirm the
guarantee and compare SE with a greedy forward-selection baseline (GE). SE
recovers feature interactions that greedy selection misses (hidden parities,
the Synergy Trap). It reaches the exact optimum in 246 of 270 runs on real
data, against 211 for GE, and returns the same explanation in every run.

## Repository structure

```
.
├── code/       implementation of IKM, SE, the greedy baseline and the experiments
├── tests/      unit tests (pytest)
└── results/    raw results of every run, figures and tables of the paper
```

- [`code/README.md`](code/README.md) maps every file to the corresponding part of the paper and lists all commands.
- [`results/README.md`](results/README.md) maps every CSV file, figure and table to the paper.

## Installation

Python ≥ 3.10 (the experiments used Python 3.14).

    pip install numpy pandas scikit-learn torch matplotlib colorama jinja2 pytest

A CUDA GPU is recommended. The code also runs on the CPU (`--device cpu`), much more slowly.

Run the tests from the root of the repository:

    pytest -q tests/

There are 31 tests. The 3 GPU tests are skipped when CUDA is not available.

## Quick example

From the `code/` folder, explain a hidden parity on 3 of 8 binary features:

```python
import torch
from blackbox import Parity
from distribution import ProductDistribution
from explainer import SpectralExplainer
from oracle import MembershipOracle
from parameters import SEParameters

d, k = 8, 3
D = ProductDistribution.uniform(d, 2)                  # uniform product distribution on {0,1}^8
f = Parity([1, 4, 6])                                  # black box with labels in {-1, +1}
params = SEParameters(epsilon=0.9, delta=0.1, k=k, d=d, c=2, kappa=0.02)

se = SpectralExplainer(MembershipOracle(f), D, params)
print(se.explain(torch.Generator().manual_seed(12)))   # [1, 4, 6]
```

To explain your own model, subclass `BlackBox` (in `blackbox.py`) and implement
`predict(X)`. It receives a `uint8` tensor of instances and must return an
`int8` tensor of labels in {−1, +1}. Use `ProductDistribution.from_data` to
build the distribution from a dataset.

## Reproducing the paper

All experiments are run from `code/` with `main.py`. The figures and tables
are produced with `displayer.py`.

| Series | Content | Paper |
|---|---|---|
| 1 | noisy parity, gap and queries against ε | Figure 1(b, c) |
| 2 | noisy parity, failure frequency against δ | Figure 2 |
| 3, 4, 5 | hidden parities, cost against d, k, c | Figure 3, Table 2 |
| 6 | Synergy Trap, err(K) against k | Figure 1(a) |
| 7 | nine OpenML datasets, k = 1, …, 5 | Tables 1, 4, 5 and Figure 4 |

```bash
cd code
python main.py --series 1 2 3 4 5 6 --gpus 0 1
python main.py --series 7 --datasets 37 15 50 43672 --runs 6 --gpus 0 1
python main.py --series 7 --datasets 179 42192 1461 24 31 --runs 6 --gpus 0 1
python main.py --series 7 --datasets 15 1461 --runs 6 --gpus 0 1
```

The figures and tables of the paper can also be regenerated directly from the
stored results, without rerunning anything. See
[`results/README.md`](results/README.md).

The experiments ran on a workstation with an Intel Xeon Gold 6252 CPU, 384 GB
of RAM and two NVIDIA Quadro RTX 8000 GPUs. On the OpenML datasets, one run of
SE at k = 5 takes between 4 and 41 minutes.

## Citation

```bibtex
@inproceedings{koriche2026learning,
  title     = {Learning Global Probabilistic Explanations},
  author    = {Koriche, Fr{\'e}d{\'e}ric and Bounia, Louenas},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=c2Ny586AOf}
}
```

## Contact

- Frédéric Koriche: frederic.koriche@cril.fr
- Louenas Bounia: bounia@lipn.univ-paris13.fr

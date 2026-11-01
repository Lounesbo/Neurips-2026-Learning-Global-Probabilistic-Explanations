# Results

Raw results, figures and tables of the experiments of *Learning Global
Probabilistic Explanations* (NeurIPS 2026), Section 5 and Appendix B.

All runs were produced by the code in [`../code`](../code) with
epsilon = 0.35 and delta = 0.1 (unless swept), kappa = 0.01, on two NVIDIA
Quadro RTX 8000 GPUs (one shard per GPU). No run reached the machine cap
(`ikm_capped` is false everywhere).

## Layout

```
results/
├── runs/        one folder per series, one CSV per GPU shard (raw results of every run)
├── figures/     figures of the paper (PDF)
└── tables/      LaTeX tables of the paper
```

## Correspondence with the paper

| Series | CSV (`runs/`) | Output | Paper |
|---|---|---|---|
| 6 Synergy Trap | `trap/` | `figures/trap.pdf` | Figure 1(a) |
| 1 epsilon sweep (noisy parity) | `eps_sweep/` | `figures/eps_gap.pdf`, `figures/eps_queries.pdf` | Figure 1(b), 1(c) |
| 2 delta sweep (noisy parity) | `delta_sweep/` | `figures/delta_sweep.pdf` | Figure 2 |
| 3, 4, 5 cost on hidden parities | `d_sweep/`, `k_sweep/`, `c_sweep/` | `figures/d_sweep.pdf`, `figures/k_sweep.pdf`, `figures/c_sweep.pdf` | Figure 3, Table 2 |
| 7 OpenML benchmarks | `datasets/` | `tables/datasets_main_k5.tex` | Table 1 |
| | | `tables/datasets_all_k5.tex` | Table 4 |
| | | `figures/datasets_<name>.pdf` | Figure 4 |
| | | column `K` of the rows with `k = 3` and `explainer = SE` | Table 5 |

Number of runs: 10 per point for series 1, 3–6, 100 per point for series 2,
and 6 per dataset and value of k for series 7. Each run holds one row for SE
and one row for GE.

## The three campaigns of series 7

`runs/datasets/` holds three campaigns, identified by their tag:

| Tag | Datasets |
|---|---|
| `20261030-020319` | Pima Indians Diabetes, Breast Cancer (WI), Tic-Tac-Toe, Heart Disease |
| `20261030-035454` | Adult, COMPAS, Bank Marketing, Mushroom, Credit-G |
| `20261031-010754` | Breast Cancer (WI), Bank Marketing (rerun) |

When a run appears in several campaigns, `displayer.py` keeps the most recent
one, so the rerun replaces the Breast Cancer (WI) and Bank Marketing rows of the
first two campaigns. All three campaigns must therefore be given together.

## Regenerating the figures and tables

From the `code/` folder (outputs go to `code/exps/plots/` and `code/exps/tables/`):

    python displayer.py "../results/runs/trap/*.csv"
    python displayer.py "../results/runs/eps_sweep/*.csv"
    python displayer.py "../results/runs/delta_sweep/*.csv"
    python displayer.py "../results/runs/d_sweep/*.csv"
    python displayer.py "../results/runs/k_sweep/*.csv"
    python displayer.py "../results/runs/c_sweep/*.csv"
    python displayer.py "../results/runs/datasets/*.csv"

Add `--no-tex` if LaTeX is not installed, and `--table 3` for the tables at k = 3.

## Main CSV columns

Each row is one explanation, by one explainer, in one run.

| Column | Meaning |
|---|---|
| `series`, `task`, `value`, `run`, `seed` | series, target (dataset or synthetic function), swept value, run index, seed |
| `explainer` | `SE` (spectral explainer, Algorithm 2) or `GE` (greedy baseline) |
| `K`, `size` | explanation returned (feature names separated by `\|`) and its size |
| `err`, `opt`, `gap`, `success` | exact err(K), optimum, err(K) − opt, and gap ≤ epsilon |
| `queries`, `walltime` | membership queries and running time (seconds) of the explainer |
| `epsilon`, `delta`, `k`, `d`, `c`, `kappa` | parameters of the run |
| `ikm_theta`, `ikm_b`, `ikm_n`, `ikm_queries` | IKM parameters: theta, number of blocks b, block size n, and (2d + 1) b n |
| `ikm_calls`, `ikm_max_calls` | weights and coefficients estimated by IKM, and the bound N = 8dc/theta² |
| `ikm_capped`, `ikm_kappa_effective` | whether the machine cap lowered n, and the resulting constant |
| `n_influential` | features kept by the pruning |
| `n_padded` (campaign `20261031-010754` only) | features added by the completion step |
| `test_accuracy` (series 7) | test accuracy of the MLP classifier f |

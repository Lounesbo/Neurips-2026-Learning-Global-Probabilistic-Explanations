# Learning Global Probabilistic Explanations — experiments

Code of the experiments of the NeurIPS 2026 paper *Learning Global Probabilistic
Explanations* (Koriche & Bounia). Every algorithm follows the paper:

| Paper | Code |
|---|---|
| Product space, Stieltjes bases (Lemmas 1–2, Stieltjes procedure) | `distribution.py` |
| (epsilon, delta) -> eps', theta, delta_0, b, n, pruning threshold | `parameters.py` |
| Membership queries (counted) | `oracle.py` |
| IKM (Algorithm 1), with shared batches per depth (Remark on shared batches) and degree cap k | `ikm.py` |
| err(K) = (1 - W(f_K)) / 2 (Lemma 4), exact or Monte Carlo | `evaluator.py` |
| SE (Algorithm 2) with completion of K up to k, GE, brute force opt | `explainer.py` |
| Classifiers (MLP queried on the GPU) and synthetic targets | `blackbox.py` |
| OpenML preprocessing (c = 2) | `dataset.py` |
| Machine cap (minutes per run -> queries) | `budget.py` |
| Driver: seeds, caps, CSV, resume | `runner.py` |
| Series of the paper | `main.py` |
| Figures and tables | `displayer.py` |

The block size is n = ceil(kappa * 64 / theta^4) with kappa = 0.01; the analysis
prescribes kappa = 1 (see the paper, Section 5 and Appendix B).

## Requirements

Python >= 3.10 (the paper used 3.14), `numpy`, `pandas`, `scikit-learn`,
`torch`, `matplotlib`, `colorama`, `pytest`. A CUDA GPU is recommended; the code
also runs on the CPU (`--device cpu`), much more slowly.

## Tests

    pytest -q tests/        # 31 tests; the 3 CUDA tests are skipped without a GPU

## Series and their place in the paper

| Series | Content | Paper |
|---|---|---|
| 1 `eps_sweep` | noisy parity, gap and queries against epsilon | Figure 1(b, c) |
| 2 `delta_sweep` | noisy parity, failure frequency against delta | Figure 2 in Appendix B.3|
| 3 `d_sweep`, 4 `k_sweep`, 5 `c_sweep` | hidden parities, cost against d, k, c | Figure 3 in Appendix B.3 |
| 6 `trap` | Synergy Trap, err against k | Figure 1(a) |
| 7 `datasets` | nine OpenML datasets, k = 1..5 | Tables 1 and 3 |

The global explanations at k = 3 (Table 5 in Appendix B.3) are read from the column `K` of
the series 7 CSV files.

## Reproducing the paper

Series 1 to 6 (default numbers of runs: 10, and 100 for series 2):

    python main.py --series 1 2 3 4 5 6 --gpus 0 1

Series 7, in the three campaigns of the paper (the seed of a run depends on the
position of its dataset in the list, so keep these lists to obtain the same
values):

    python main.py --series 7 --datasets 37 15 50 43672 --runs 6 --gpus 0 1
    python main.py --series 7 --datasets 179 42192 1461 24 31 --runs 6 --gpus 0 1
    python main.py --series 7 --datasets 15 1461 --runs 6 --gpus 0 1

Figures and tables (written to `exps/plots/` and `exps/tables/`):

    python displayer.py "exps/eps_sweep/*.csv"      # eps_gap, eps_queries
    python displayer.py "exps/delta_sweep/*.csv"    # delta_sweep
    python displayer.py "exps/d_sweep/*.csv"        # d_sweep (same for k_sweep, c_sweep)
    python displayer.py "exps/trap/*.csv"           # trap
    python displayer.py "exps/datasets/*.csv"       # datasets_<name>, datasets_main_k5.tex, datasets_all_k5.tex

When several campaigns hold the same run, the displayer takes it from the most
recent one. Add `--no-tex` if LaTeX is not installed, and `--table 3` for the
tables at k = 3.

## Other commands

    python main.py --inspect                                   # dataset statistics after binarization
    python main.py --series 1 2 --calibrate                    # once: throughputs -> exps/rates.json
    python main.py --series 1 2 --device cuda:0 --shard 0/2    # by hand: GPU 0, even runs
    python main.py --series 1 2 --device cuda:1 --shard 1/2    #          GPU 1, odd runs
    python main.py --series 7 --minutes 60                     # larger machine cap
    python main.py --series 7 --runs 6 --gpus 0 1 --tag <tag>  # resume the campaign <tag>

## Output

Results go to `exps/<series>/`, one CSV per shard, rewritten after every run;
logs go to `exps/logs/`. Rerunning with the tag of an interrupted campaign skips
its finished runs. Run from a local folder: a synchronized folder (Google Drive,
OneDrive) may invalidate open files during a long campaign.

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackbox import NoisyParity, Parity
from distribution import ProductDistribution
from evaluator import Evaluator
from explainer import BruteForceExplainer, GreedyExplainer, SpectralExplainer
from oracle import MembershipOracle
from parameters import SEParameters
from test_core import Table, exact_spectrum, gen, random_product


def test_exact_error_matches_spectrum():
    """err(K) = (1 - sum of f(s)^2 over supp(s) in K) / 2, Lemma 4."""
    rng = np.random.default_rng(10)
    D = random_product(4, 3, rng)
    f = Table(D.cards, rng)
    S, coefficients = exact_spectrum(D, f)
    S = S.numpy()
    ev = Evaluator(f, D)
    for K in [(), (0,), (1, 3), (0, 1, 2), (0, 1, 2, 3)]:
        inside = ~(S[:, [i for i in range(4) if i not in K]] != 0).any(axis=1)
        assert np.isclose(ev.error(K), 0.5 * (1 - np.sum(coefficients[inside] ** 2)))


def test_monte_carlo_error_matches_exact():
    rng = np.random.default_rng(11)
    D = random_product(5, 3, rng)
    f = Table(D.cards, rng)
    exact = Evaluator(f, D)
    mc = Evaluator(f, D, n_samples=400_000, exact_limit=0, chunk_size=1 << 16)
    for K in [(0,), (1, 2), (0, 2, 4)]:
        assert abs(exact.error(K) - mc.error(K)) < 4 / (2 * np.sqrt(400_000))


def test_brute_force_on_parity():
    D = ProductDistribution.uniform(8, 2)
    bf = BruteForceExplainer(Evaluator(Parity([1, 4, 6]), D), k=3)
    assert bf.explain() == [1, 4, 6] and np.isclose(bf.opt, 0.0)


def test_spectral_explainer_finds_the_parity():
    d, k, support = 8, 3, [1, 4, 6]
    D = ProductDistribution.uniform(d, 2)
    P = SEParameters(epsilon=0.9, delta=0.1, k=k, d=d, c=2, kappa=0.02)
    se = SpectralExplainer(MembershipOracle(Parity(support)), D, P)
    assert se.explain(gen(12)) == support
    assert se.influential == support


def test_spectral_explainer_completes_up_to_k():
    # Only the two parity features survive the pruning; K is completed to k = 4
    # by decreasing influence, which keeps err(K) = 0.
    d, k, support = 8, 4, [1, 4]
    D = ProductDistribution.uniform(d, 2)
    P = SEParameters(epsilon=0.9, delta=0.1, k=k, d=d, c=2, kappa=0.02)
    se = SpectralExplainer(MembershipOracle(Parity(support)), D, P)
    K = se.explain(gen(12))
    assert se.influential == support and se.n_padded == 2
    assert len(K) == k and set(support) <= set(K)
    assert np.isclose(Evaluator(Parity(support), D).error(K), 0.0)


def test_greedy_is_blind_to_parity():
    """
    No subset of S carries any signal, so greedy picks at random among ties
    and succeeds with probability about k! / d^k only (here about 1 / 4000).
    """
    d, k, support = 30, 3, [3, 11, 25]
    D = ProductDistribution.uniform(d, 2)
    f = Parity(support)
    ev = Evaluator(f, D, n_samples=20_000)
    successes = 0
    for seed in range(10):
        K = GreedyExplainer(MembershipOracle(f), D, k=k).explain(gen(seed))
        successes += ev.error(K) < 0.1
    assert successes <= 1


def test_noisy_parity_opt():
    d, eta = 12, 0.1
    f = NoisyParity([0, 5, 9], eta=eta, d=d, seed=3)
    D = ProductDistribution.uniform(d, 2)
    bf = BruteForceExplainer(Evaluator(f, D), k=3)
    assert bf.explain() == [0, 5, 9]
    # 2 eta (1 - eta) is the large-space limit; on 4096 points the realized
    # flip rate deviates by O(1/sqrt(|Omega|)), and the experiments use the
    # exact opt given by the brute force anyway.
    assert abs(bf.opt - 2 * eta * (1 - eta)) < 0.03

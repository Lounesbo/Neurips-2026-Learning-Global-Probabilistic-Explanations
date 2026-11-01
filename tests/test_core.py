import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackbox import BlackBox, Classifier, NoisyParity, Parity
from distribution import Basis, ProductDistribution
from ikm import IKM
from oracle import MembershipOracle
from parameters import IKMParameters, SEParameters


def gen(seed):
    return torch.Generator().manual_seed(seed)


class Table(BlackBox):
    """Arbitrary f on a small space, given by its table of values."""

    def __init__(self, cards, rng):
        self.cards = [int(c) for c in cards]
        self.values = torch.tensor(rng.choice([-1, 1], size=int(np.prod(cards))), dtype=torch.int8)

    def predict(self, X):
        index = torch.zeros(X.shape[0], dtype=torch.long)
        for i, c in enumerate(self.cards):
            index = index * c + X[:, i].long()
        return self.values[index]


def exact_spectrum(D: ProductDistribution, f: BlackBox):
    """All coordinates of S and the exact coefficients of f, by enumeration."""
    X = D.enumerate()
    S = X.clone()  # same index set: prod {0..c_i - 1}
    coefficients = D.phi(S, X, dtype=torch.float64) @ (D.probabilities(X) * f.predict(X).double())
    return S, coefficients.numpy()


def random_product(d, c, rng, **kwargs):
    marginals = []
    for _ in range(d):
        p = rng.dirichlet(np.ones(c)) + 0.05
        marginals.append(p / p.sum())
    return ProductDistribution(marginals, **kwargs)


# -- Bases ---------------------------------------------------------------------

@pytest.mark.parametrize("c", [2, 3, 4, 6, 8])
def test_basis_is_orthonormal(c):
    rng = np.random.default_rng(c)
    mu = rng.dirichlet(np.ones(c)) + 1e-3
    basis = Basis(mu / mu.sum())
    assert np.allclose(basis.phi[0], 1.0)
    assert basis.leakage < 1e-10


def test_parseval_on_random_function():
    rng = np.random.default_rng(1)
    D = random_product(4, 3, rng)
    _, coefficients = exact_spectrum(D, Table(D.cards, rng))
    assert np.isclose(np.sum(coefficients ** 2), 1.0)


def test_sampler_matches_marginals():
    rng = np.random.default_rng(2)
    D = random_product(3, 4, rng)
    X = D.sample(200_000, gen(2)).numpy()
    for i, basis in enumerate(D.bases):
        freq = np.bincount(X[:, i], minlength=4) / X.shape[0]
        assert np.allclose(freq, basis.mu, atol=5e-3)


def test_mixed_alphabets():
    D = ProductDistribution([np.array([0.3, 0.7]), np.array([0.2, 0.3, 0.5])])
    X = D.sample(100_000, gen(3)).numpy()
    assert X[:, 0].max() == 1 and X[:, 1].max() == 2


# -- Parameters ----------------------------------------------------------------

def test_se_parameters_follow_the_paper():
    P = SEParameters(epsilon=0.35, delta=0.1, k=3, d=20, c=2)
    assert np.isclose(P.epsilon_prime, 0.1)
    assert np.isclose(P.theta, 0.1 / 2 ** 1.5)
    assert np.isclose(P.prune_threshold, 0.1 / 3)
    ikm = P.ikm
    assert np.isclose(ikm.delta0, ikm.theta ** 2 * 0.1 / (8 * 20 * 2))
    assert ikm.b == int(np.ceil(8 * np.log(1 / ikm.delta0)))
    # theta^2 = 0.01 / 8, so 64 / theta^4 = 64 * 64 / 1e-4 exactly.
    assert ikm.n == 40_960_000


# -- Black boxes ---------------------------------------------------------------

def test_torch_mlp_matches_sklearn():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 2, size=(3000, 12)).astype(np.uint8)
    Y = np.where((X[:, 0] ^ X[:, 3]) | X[:, 7], 1, -1).astype(np.int8)
    clf = Classifier("Neural Network", seed=0, max_iter=300)
    clf.train(X, Y)
    Z = rng.integers(0, 2, size=(20_000, 12)).astype(np.uint8)
    agree = np.mean(clf.predict(torch.from_numpy(Z)).numpy() == clf.model.predict(Z))
    assert agree > 0.9999


def test_noisy_parity_flip_rate():
    d, eta = 30, 0.1
    f = NoisyParity([0, 1, 2], eta=eta, d=d, seed=7)
    X = ProductDistribution.uniform(d, 2).sample(200_000, gen(8))
    assert abs(f.flipped(X).float().mean().item() - eta) < 5e-3
    assert torch.equal(f.predict(X[:1000]), f.predict(X[:1000]))


# -- IKM -----------------------------------------------------------------------

def test_weights_are_unbiased():
    """Weight(sigma) against the exact cone weight, for every prefix of every length."""
    rng = np.random.default_rng(3)
    D = random_product(4, 3, rng)
    f = Table(D.cards, rng)
    S, coefficients = exact_spectrum(D, f)
    S = S.numpy()
    params = IKMParameters(theta=1.0, delta=0.5, d=D.d, c=D.c, kappa=1.0)
    ikm = IKM(MembershipOracle(f), D, params)
    ikm.params = type("P", (), {"b": 9, "n": 40_000})()  # direct control of the budget
    for p in range(1, D.d + 1):
        prefixes = np.unique(S[:, :p], axis=0)
        exact = np.array([np.sum(coefficients[(S[:, :p] == s).all(axis=1)] ** 2) for s in prefixes])
        estimated = ikm._weights(torch.from_numpy(prefixes), gen(p)).numpy()
        assert np.allclose(estimated, exact, atol=0.03), p


def test_coefficients_are_unbiased():
    rng = np.random.default_rng(5)
    D = random_product(3, 3, rng)
    f = Table(D.cards, rng)
    S, coefficients = exact_spectrum(D, f)
    params = IKMParameters(theta=1.0, delta=0.5, d=D.d, c=D.c)
    ikm = IKM(MembershipOracle(f), D, params)
    ikm.params = type("P", (), {"b": 9, "n": 40_000})()
    assert np.allclose(ikm._coefficients(S, gen(5)).numpy(), coefficients, atol=0.03)


@pytest.mark.parametrize("c", [2, 4])
def test_ikm_recovers_a_parity(c):
    d, support = 12, [2, 5, 9]
    D = ProductDistribution.uniform(d, c)
    oracle = MembershipOracle(Parity(support))
    params = IKMParameters(theta=0.25, delta=0.1, d=d, c=c, kappa=1.0)
    ikm = IKM(oracle, D, params, max_degree=3)
    ikm.learn(gen(4))

    assert not ikm.overflow
    supports = {tuple(np.nonzero(s)[0]) for s in ikm.coordinates}
    assert supports == {tuple(support)}
    # Completeness only covers |f(s)| >= theta: for c = 4, (-1)^{x_i} spreads
    # over several phi_{i,j}, and coefficients below theta may be missed.
    energy = np.sum(ikm.coefficients ** 2)
    assert (np.isclose(energy, 1.0, atol=0.05) if c == 2 else 0.85 < energy < 1.05)
    assert ikm.n_queries == params.queries
    assert ikm.n_calls <= params.max_calls


def test_degree_cap_removes_high_degree_coordinates():
    d = 8
    D = ProductDistribution.uniform(d, 2)
    params = IKMParameters(theta=0.25, delta=0.1, d=d, c=2)
    ikm = IKM(MembershipOracle(Parity([0, 1, 2, 3])), D, params, max_degree=3)
    ikm.learn(gen(5))
    assert ikm.coordinates.shape[0] == 0 and not ikm.overflow


def test_chunking_does_not_change_estimates():
    """Small memory forces both chunking regimes; results must match the unchunked run."""
    D = ProductDistribution.uniform(6, 2)
    f = Parity([1, 4])
    params = IKMParameters(theta=0.5, delta=0.1, d=6, c=2)
    reference = IKM(MembershipOracle(f), D, params, max_degree=2)
    reference.learn(gen(6))
    for memory in [params.n * 3, params.n // 7]:
        ikm = IKM(MembershipOracle(f), D, params, max_degree=2, memory=memory)
        ikm.learn(gen(6))
        assert {tuple(np.nonzero(s)[0]) for s in ikm.coordinates} == {(1, 4)}
        assert ikm.n_queries == reference.n_queries


def test_phi_is_the_product_of_univariate_bases():
    rng = np.random.default_rng(3)
    cards = [2, 3, 4, 2, 3]
    D = ProductDistribution([rng.dirichlet(np.ones(c)) for c in cards])
    C = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 2, 3, 1], [1, 1, 0, 1]], dtype=torch.uint8)
    X = torch.tensor([[int(rng.integers(c)) for c in cards[:4]] for _ in range(50)], dtype=torch.uint8)
    expected = torch.ones((4, 50), dtype=torch.float64)
    for r in range(4):
        for i in range(4):
            expected[r] *= D._phi[i, int(C[r, i])][X[:, i].long()]
    assert torch.allclose(D.phi(C, X, dtype=torch.float64), expected)

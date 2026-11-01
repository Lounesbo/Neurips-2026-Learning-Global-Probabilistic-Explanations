# -----------------------------------------------------------------------------
#
# Explainers
#
# Every explainer returns a subset K of at most k features. The spectral and
# greedy explainers access f through the same MembershipOracle, so that their
# query counts are comparable; the brute force explainer evaluates f on all of
# Omega and serves only to compute opt on small spaces.
#
# -----------------------------------------------------------------------------

import itertools
import math
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import Dict, List

import numpy as np
import torch

from distribution import ProductDistribution
from evaluator import Evaluator
from ikm import IKM
from oracle import MembershipOracle
from parameters import SEParameters


class Explainer(ABC):
    def __init__(self, k: int):
        self.k = k
        self.explanation: List[int] = []
        self.walltime = 0.0
        self.n_queries = 0

    @abstractmethod
    def explain(self, generator: torch.Generator) -> List[int]:
        pass

    def stats(self) -> Dict[str, object]:
        """Columns of the CSV specific to this explainer."""
        return {}


# -- Spectral explainer (Algorithm 2) --------------------------------------------

class SpectralExplainer(Explainer):
    def __init__(self, oracle: MembershipOracle, distribution: ProductDistribution,
                 params: SEParameters, chunk: int = 1 << 14):
        super().__init__(params.k)
        self.oracle = oracle
        self.D = distribution
        self.params = params
        self.chunk = chunk  # candidate subsets scored at once
        self.ikm = IKM(oracle, distribution, params.ikm, max_degree=params.k)
        self.influences = np.zeros(distribution.d)
        self.influential: List[int] = []
        self.energy = 0.0
        self.n_candidates = 0
        self.n_padded = 0

    def explain(self, generator: torch.Generator) -> List[int]:
        start = timer()
        queries = self.oracle.n_queries

        # Spectral recovery
        self.ikm.learn(generator)
        coordinates, coefficients = self.ikm.coordinates, self.ikm.coefficients
        supports = coordinates != 0                       # (L0, d)
        energies = coefficients ** 2
        # IKM already caps the degree at k; the filter states R_i^k explicitly.
        low = supports.sum(axis=1) <= self.k

        # Feature pruning: Inf_i^k[g] = sum of g(s)^2 over s in R_i^k
        self.influences = energies[low] @ supports[low]
        self.influential = [int(i) for i in np.nonzero(self.influences >= self.params.prune_threshold)[0]]

        # Subset selection over I. Coordinates are kept when their support is
        # included in I and has at most k features, and are merged by support.
        I = np.array(self.influential, dtype=int)
        inside = low & ~supports[:, np.setdiff1d(np.arange(self.D.d), I)].any(axis=1)
        by_support: Dict[tuple, float] = {}
        for row, e in zip(supports[inside][:, I], energies[inside]):
            key = tuple(np.nonzero(row)[0])
            by_support[key] = by_support.get(key, 0.0) + float(e)

        self.explanation, self.energy = self._search(by_support, len(I))
        self.explanation = [int(I[j]) for j in self.explanation]

        # Completion. err is monotone (K in K' implies W(f_K) <= W(f_K')), so
        # adding features never increases err(K) and keeps the guarantee. When
        # fewer than k features survive the pruning, K is completed up to k by
        # decreasing estimated influence Inf_i^k[g], at no extra query.
        missing = self.k - len(self.explanation)
        if missing > 0:
            rest = [i for i in np.argsort(-self.influences, kind="stable") if i not in self.explanation]
            self.explanation = sorted(self.explanation + [int(i) for i in rest[:missing]])
        self.n_padded = max(missing, 0)

        self.n_queries = self.oracle.n_queries - queries
        self.walltime = timer() - start
        return self.explanation

    def _search(self, by_support: Dict[tuple, float], m: int):
        """
        argmax of W(g_K) over K in I with |K| <= k. Energies are nonnegative,
        so W(g_K) is monotone in K and subsets of size min(k, |I|) suffice.
        W(g_K) sums the energies of the supports included in K: with A the
        (supports x I) incidence matrix, U is in K iff A_U . (1 - 1_K) = 0.
        """
        size = min(self.k, m)
        self.n_candidates = math.comb(m, size)
        if size == 0 or not by_support:
            return list(range(size)), by_support.get((), 0.0)
        A = np.zeros((len(by_support), m))
        for row, key in enumerate(by_support):
            A[row, list(key)] = 1.0
        e = np.array(list(by_support.values()))

        best, best_energy = None, -1.0
        combos = itertools.combinations(range(m), size)
        while True:
            block = list(itertools.islice(combos, self.chunk))
            if not block:
                break
            outside = np.ones((len(block), m))
            outside[np.arange(len(block))[:, None], np.array(block)] = 0.0
            W = ((A @ outside.T) == 0).T @ e
            j = int(np.argmax(W))
            if W[j] > best_energy:
                best, best_energy = list(block[j]), float(W[j])
        return best, best_energy

    def stats(self) -> Dict[str, object]:
        ikm = self.ikm
        return {
            "ikm_calls": ikm.n_calls,
            "ikm_weight_calls": ikm.n_weight_calls,
            "ikm_coefficient_calls": ikm.n_coefficient_calls,
            "ikm_max_retained": max(ikm.retained_per_depth, default=0),
            "ikm_overflow": ikm.overflow,
            "ikm_terms": int(ikm.coordinates.shape[0]),
            "ikm_walltime": ikm.walltime,
            "n_influential": len(self.influential),
            "n_candidates": self.n_candidates,
            "n_padded": self.n_padded,
            "energy_g": self.energy,
        }


# -- Greedy explainer (forward selection, after Bassan et al.) ---------------------

class GreedyExplainer(Explainer):
    """
    Starts from K = {} and adds the feature that most decreases the estimated
    err(K), until no candidate improves it or |K| = k. Each step draws one
    fresh sample X, queried once, and each estimate draws its own Z.

    err(K) itself is re-estimated at every step, on the sample of that step.
    Carrying over the best error of the previous step would compare the new
    candidates with a minimum of noisy estimates, which is biased downwards,
    and greedy would stop too early.

    Z must not be shared between candidates. With a common Z, any feature i
    that f ignores yields exactly the current error, since z and z with x_i
    give the same label, while the relevant features fluctuate around it: the
    argmin then selects relevant features by their noise alone. This is a
    hidden influence test, and it lets greedy recover parities, which forward
    selection on err(K) cannot do.
    """

    def __init__(self, oracle: MembershipOracle, distribution: ProductDistribution,
                 k: int, n_samples: int = 5000, stop_on_no_improvement: bool = True):
        super().__init__(k)
        self.oracle = oracle
        self.D = distribution
        self.n_samples = n_samples
        self.stop_on_no_improvement = stop_on_no_improvement

    def explain(self, generator: torch.Generator) -> List[int]:
        start = timer()
        queries = self.oracle.n_queries
        K: List[int] = []
        for _ in range(self.k):
            X = self.D.sample(self.n_samples, generator)
            fX = self.oracle(X)
            current = self._error(fX, X, K, generator)
            best_i, best_err = -1, float("inf")
            for i in range(self.D.d):
                if i in K:
                    continue
                e = self._error(fX, X, K + [i], generator)
                if e < best_err:
                    best_i, best_err = i, e
            if best_i < 0 or (self.stop_on_no_improvement and best_err >= current):
                break
            K.append(best_i)
        self.explanation = sorted(K)
        self.n_queries = self.oracle.n_queries - queries
        self.walltime = timer() - start
        return self.explanation

    def _error(self, fX: torch.Tensor, X: torch.Tensor, K: List[int], generator: torch.Generator) -> float:
        Z = self.D.sample(self.n_samples, generator)
        Z[:, K] = X[:, K]
        return float((fX != self.oracle(Z)).float().mean())


# -- Brute force (opt, on small spaces) ---------------------------------------------

class BruteForceExplainer(Explainer):
    """Exact argmax of W(f_K) over |K| = min(k, d), by enumeration of Omega."""

    def __init__(self, evaluator: Evaluator, k: int):
        super().__init__(k)
        if not evaluator.is_exact:
            raise ValueError("Brute force needs an evaluator in exact mode")
        self.evaluator = evaluator
        self.energy = 0.0

    def explain(self, generator: torch.Generator = None) -> List[int]:
        start = timer()
        d = self.evaluator.D.d
        best, best_energy = [], -1.0
        for K in itertools.combinations(range(d), min(self.k, d)):
            w = self.evaluator.energy(K)
            if w > best_energy:
                best, best_energy = list(K), w
        self.explanation, self.energy = best, best_energy
        self.walltime = timer() - start
        return self.explanation

    @property
    def opt(self) -> float:
        return 0.5 * (1.0 - self.energy)

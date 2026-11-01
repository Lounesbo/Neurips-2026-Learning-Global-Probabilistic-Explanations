# -----------------------------------------------------------------------------
#
# Evaluator: the global relevance error of the paper, and nothing else
#
#   err(K) = P_{x ~ D, z ~ D(. | z_K = x_K)}[f(x) != f(z)] = (1 - W(f_K)) / 2
#
# Since D is a product distribution, z ~ D(. | z_K = x_K) is x on K and a fresh
# draw from the marginals elsewhere.
#
# Exact mode, when |Omega| <= exact_limit: f is evaluated once on all of Omega,
# f_K is the D-weighted mean of f over each class of x_K, and W(f_K) = E[f_K^2].
# Otherwise Monte Carlo with a fixed budget: the indicator is bounded, so the
# standard error is at most 1 / (2 sqrt(n_samples)) whatever the dimension. All
# subsets evaluated by one Evaluator share the same draws.
#
# The evaluator calls the black box directly, not through an oracle: measuring
# err(K) is not part of explaining, and must not show in the query counts.
#
# -----------------------------------------------------------------------------

from typing import Iterable

import numpy as np
import torch

from blackbox import BlackBox
from distribution import ProductDistribution


class Evaluator:
    def __init__(self, blackbox: BlackBox, distribution: ProductDistribution,
                 n_samples: int = 1_000_000, exact_limit: int = 1 << 22, seed: int = 0,
                 chunk_size: int = 1 << 20):
        self.f = blackbox
        self.D = distribution
        self.n_samples = n_samples
        self.seed = seed
        self.chunk_size = chunk_size
        self.size = int(np.prod(self.D.cards.astype(float)))
        self.is_exact = self.size <= exact_limit
        self._cache = None

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.f.predict(X[s:s + self.chunk_size])
                          for s in range(0, X.shape[0], self.chunk_size)])

    def _enumerate(self):
        if self._cache is None:
            points = self.D.enumerate()
            probs = self.D.probabilities(points)
            weights = probs * self._predict(points).to(torch.float64)
            self._cache = (points, probs, weights)
        return self._cache

    def energy(self, K: Iterable[int]) -> float:
        """W(f_K), exactly. Only available in exact mode."""
        if not self.is_exact:
            raise RuntimeError("Exact energies need |Omega| <= exact_limit")
        points, probs, weights = self._enumerate()
        K = sorted(K)
        if not K:
            return float(weights.sum() ** 2)
        groups = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        for i in K:  # row-major index of x_K
            groups = groups * int(self.D.cards[i]) + points[:, i].long()
        size = int(np.prod(self.D.cards[K]))
        num = torch.zeros(size, dtype=torch.float64, device=points.device).index_add_(0, groups, weights)
        den = torch.zeros(size, dtype=torch.float64, device=points.device).index_add_(0, groups, probs)
        mask = den > 0
        return float((num[mask] ** 2 / den[mask]).sum())

    def error(self, K: Iterable[int]) -> float:
        K = sorted(K)
        if self.is_exact:
            return 0.5 * (1.0 - self.energy(K))
        generator = torch.Generator(device=self.D.device).manual_seed(self.seed)
        disagreements = 0
        for start in range(0, self.n_samples, self.chunk_size):
            L = min(self.chunk_size, self.n_samples - start)
            X = self.D.sample(L, generator)
            Z = self.D.sample(L, generator)
            Z[:, K] = X[:, K]
            disagreements += int((self.f.predict(X) != self.f.predict(Z)).sum())
        return disagreements / self.n_samples

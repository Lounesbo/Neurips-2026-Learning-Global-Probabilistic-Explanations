# -----------------------------------------------------------------------------
#
# Integer Kushilevitz--Mansour (Algorithm 1)
#
# The search is run breadth-first, one depth at a time, with two changes to the
# letter of Algorithm 1 that leave the guarantee of SE untouched:
#
# 1. Shared batches. The queries made by Weight(sigma), f(U, V) and f(U', V),
#    depend on the length p of sigma but not on sigma itself: only the factor
#    phi_sigma does. All prefixes of length p are therefore estimated on one
#    fresh batch of b blocks of n triples (U, U', V). Each estimate has exactly
#    the law of Lemma 2, and the set of prefixes examined at depth p depends on
#    the batches of earlier depths only, so that, conditionally on the past,
#    each estimate is accurate with probability at least 1 - delta_0 and the
#    union bound of Proposition A.1 goes through unchanged. Leaves share one
#    batch for Coefficient in the same way. The number of queries becomes
#    (2d + 1) b n instead of up to 3 b n per estimate.
#
# 2. Degree cap. With max_degree = k, a prefix whose support already holds k
#    features is only extended by 0. SE discards every coordinate with more
#    than k features in its support, and Lemmas 5 and 6 only involve such
#    coordinates, so nothing SE uses is lost; completeness and soundness hold
#    on the coordinates with support size at most k.
#
# Weight(lambda) = E[f(V)^2] = 1 for a classifier, so the root is counted as an
# estimate but costs no query.
#
# If the number of estimates is about to exceed N = 8dc / theta^2, the success
# event of Proposition A.1 has failed: the search stops, `overflow` is set, and
# an empty hypothesis is returned. This also bounds the memory of the search.
#
# Values are computed in the dtype of the distribution (float32 by default) and
# accumulated per block in float64.
#
# -----------------------------------------------------------------------------

from timeit import default_timer as timer
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from distribution import ProductDistribution
from oracle import MembershipOracle
from parameters import IKMParameters


class IKM:
    def __init__(
        self,
        oracle: MembershipOracle,
        distribution: ProductDistribution,
        params: IKMParameters,
        max_degree: Optional[int] = None,
        memory: int = 1 << 25,
        max_chunk: int = 1 << 22,
    ):
        if params.d != distribution.d or params.c != distribution.c:
            raise ValueError("The parameters do not match the distribution")
        self.oracle = oracle
        self.D = distribution
        self.params = params
        self.max_degree = distribution.d if max_degree is None else max_degree
        # Largest number of entries of a (prefixes x samples) matrix at once, and
        # largest number of samples drawn at once: the sampler builds float64
        # temporaries of a few times (samples x d) entries, which reached several
        # GB with 2^25 samples and fragmented the GPU cache.
        self.memory = memory
        self.max_chunk = max_chunk
        self._reset()

    def _reset(self) -> None:
        self.coordinates = np.zeros((0, self.D.d), dtype=np.uint8)
        self.coefficients = np.zeros(0)
        self.n_weight_calls = 0
        self.n_coefficient_calls = 0
        self.n_queries = 0
        self.retained_per_depth: List[int] = []
        self.overflow = False
        self.walltime = 0.0

    # -- Public interface ------------------------------------------------------

    @property
    def n_calls(self) -> int:
        return self.n_weight_calls + self.n_coefficient_calls

    @property
    def hypothesis(self) -> Dict[Tuple[int, ...], float]:
        return {tuple(int(v) for v in s): float(g) for s, g in zip(self.coordinates, self.coefficients)}

    def learn(self, generator: torch.Generator) -> Dict[Tuple[int, ...], float]:
        self._reset()
        start_time = timer()
        start_queries = self.oracle.n_queries
        threshold = self.params.threshold

        prefixes = torch.zeros((1, 0), dtype=torch.uint8, device=self.D.device)
        self.n_weight_calls = 1  # the root, whose weight is exactly 1

        for p in range(1, self.D.d + 1):
            children = self._children(prefixes, p)
            if not self._budget(children.shape[0]):
                break
            self.n_weight_calls += children.shape[0]
            weights = self._weights(children, generator)
            prefixes = children[weights >= threshold]
            self.retained_per_depth.append(int(prefixes.shape[0]))
            if prefixes.shape[0] == 0:
                break

        if not self.overflow and prefixes.shape[1] == self.D.d and prefixes.shape[0] > 0:
            if self._budget(prefixes.shape[0]):
                self.n_coefficient_calls = int(prefixes.shape[0])
                self.coefficients = self._coefficients(prefixes, generator).cpu().numpy()
                self.coordinates = prefixes.cpu().numpy()

        self.n_queries = self.oracle.n_queries - start_queries
        self.walltime = timer() - start_time
        return self.hypothesis

    # -- Search ----------------------------------------------------------------

    def _children(self, parents: torch.Tensor, p: int) -> torch.Tensor:
        """All extensions of the parents by a value of feature p, capped in degree."""
        c_p = int(self.D.cards[p - 1])
        m = parents.shape[0]
        device = parents.device
        values = torch.arange(c_p, dtype=torch.uint8, device=device).repeat(m)
        children = torch.cat([parents.repeat_interleave(c_p, dim=0), values[:, None]], dim=1)
        degrees = (parents != 0).sum(dim=1)
        allowed = (values == 0) | (degrees < self.max_degree).repeat_interleave(c_p)
        return children[allowed]

    def _budget(self, extra: int) -> bool:
        if self.n_calls + extra > self.params.max_calls:
            self.overflow = True
            self.coordinates = np.zeros((0, self.D.d), dtype=np.uint8)
            self.coefficients = np.zeros(0)
            return False
        return True

    # -- Estimators ------------------------------------------------------------

    def _weights(self, prefixes: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Weight(sigma) for every prefix of length p, on one shared batch."""
        p = prefixes.shape[1]
        head, tail = range(p), range(p, self.D.d)

        support = self.D.support(prefixes)

        def values(L: int) -> torch.Tensor:
            U = self.D.sample(L, generator, head)
            U2 = self.D.sample(L, generator, head)
            V = self.D.sample(L, generator, tail)
            F = self.oracle(torch.cat([U, V], dim=1)).to(self.D.dtype)
            F2 = self.oracle(torch.cat([U2, V], dim=1)).to(self.D.dtype)
            return (self.D.phi(prefixes, U, support=support) * F) * (self.D.phi(prefixes, U2, support=support) * F2)

        return self._median_of_means(prefixes.shape[0], values)

    def _coefficients(self, coordinates: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Coefficient(s) for every leaf, on one shared batch."""

        support = self.D.support(coordinates)

        def values(L: int) -> torch.Tensor:
            Z = self.D.sample(L, generator)
            F = self.oracle(Z).to(self.D.dtype)
            return self.D.phi(coordinates, Z, support=support) * F

        return self._median_of_means(coordinates.shape[0], values)

    def _median_of_means(self, m: int, values: Callable[[int], torch.Tensor]) -> torch.Tensor:
        """
        Median over b blocks of the means over n samples, for m targets at once.
        Samples are drawn in chunks that hold whole blocks when n is small, and
        that never straddle two blocks when n is large, so that every chunk is
        reduced by a plain sum.
        """
        b, n = self.params.b, self.params.n
        L = max(1, min(self.memory // max(m, 1), self.max_chunk))
        sums = torch.zeros((m, b), dtype=torch.float64, device=self.D.device)
        if n <= L:
            per = max(1, L // n)
            for q in range(0, b, per):
                nb = min(per, b - q)
                sums[:, q:q + nb] += values(nb * n).view(m, nb, n).sum(dim=-1, dtype=torch.float64)
        else:
            for q in range(b):
                for offset in range(0, n, L):
                    sums[:, q] += values(min(L, n - offset)).sum(dim=-1, dtype=torch.float64)
        means = (sums / n).sort(dim=1).values
        return (means[:, (b - 1) // 2] + means[:, b // 2]) / 2

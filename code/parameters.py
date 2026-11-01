# -----------------------------------------------------------------------------
#
# Parameters derived from the theory
#
# The only free inputs are (epsilon, delta), read off the paper, the dimensions
# (d, k, c), read off the problem, and one scale constant kappa on the block
# size. Every threshold the algorithms use is computed here and nowhere else,
# and every value is exported to the CSV through as_dict().
#
#   SE  (Algorithm 2): eps' = 2 eps / 7, theta = eps' / c^{k/2}, prune at eps'/k
#   IKM (Algorithm 1): delta_0 = theta^2 delta / (8 d c),
#                      b = ceil(8 ln(1/delta_0)), n = ceil(kappa * 64 / theta^4)
#
# kappa = 1 is the theory. kappa < 1 is the calibrated version: a single value,
# fixed once on the parity benchmark, then used everywhere.
#
# Machine cap. Since theta = eps' / c^{k/2}, the block size grows as c^{2k} /
# eps'^4 and can exceed what a machine runs in reasonable time. max_queries
# bounds the number of queries of one IKM run, (2d + 1) b n with shared batches,
# by lowering n only: b, hence the dependence on delta, is never touched. A
# capped run is flagged, and its effective kappa = n theta^4 / 64 is exported,
# so that every point of a figure says whether it was run at the calibrated
# constant or below it.
#
# -----------------------------------------------------------------------------

import math
from dataclasses import dataclass, asdict
from typing import Optional


def _ceil(x: float) -> int:
    """Ceiling that ignores floating point noise: 409600.0000000001 -> 409600."""
    return math.ceil(x - 1e-9 * max(1.0, abs(x)))


@dataclass(frozen=True)
class IKMParameters:
    theta: float
    delta: float
    d: int
    c: int
    kappa: float = 1.0
    max_queries: Optional[int] = None

    def __post_init__(self):
        if not 0 < self.theta <= 1:
            raise ValueError("theta must lie in (0, 1]")
        if not 0 < self.delta < 1:
            raise ValueError("delta must lie in (0, 1)")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")

    @property
    def delta0(self) -> float:
        return self.theta ** 2 * self.delta / (8 * self.d * self.c)

    @property
    def b(self) -> int:
        return _ceil(8 * math.log(1 / self.delta0))

    @property
    def n_uncapped(self) -> int:
        return max(1, _ceil(self.kappa * 64 / self.theta ** 4))

    @property
    def n(self) -> int:
        if self.max_queries is None:
            return self.n_uncapped
        n_max = max(1, self.max_queries // ((2 * self.d + 1) * self.b))
        return min(self.n_uncapped, n_max)

    @property
    def capped(self) -> bool:
        return self.n < self.n_uncapped

    @property
    def kappa_effective(self) -> float:
        return self.n * self.theta ** 4 / 64

    @property
    def queries(self) -> int:
        """Queries of a full run with shared batches: (2d + 1) b n."""
        return (2 * self.d + 1) * self.b * self.n

    @property
    def threshold(self) -> float:
        """A prefix is retained when its estimated weight reaches theta^2 / 2."""
        return self.theta ** 2 / 2

    @property
    def max_calls(self) -> float:
        """N = 8dc / theta^2: on the success event, IKM makes at most N estimates (Prop. A.1)."""
        return 8 * self.d * self.c / self.theta ** 2

    @property
    def max_terms(self) -> float:
        """L_0(g) <= 4 / theta^2 (Theorem 1)."""
        return 4 / self.theta ** 2

    def as_dict(self) -> dict:
        out = asdict(self)
        out.update(delta0=self.delta0, b=self.b, n=self.n, n_uncapped=self.n_uncapped,
                   capped=self.capped, kappa_effective=self.kappa_effective,
                   queries=self.queries, threshold=self.threshold,
                   max_calls=self.max_calls, max_terms=self.max_terms)
        return out


@dataclass(frozen=True)
class SEParameters:
    epsilon: float
    delta: float
    k: int
    d: int
    c: int
    kappa: float = 1.0
    max_queries: Optional[int] = None

    def __post_init__(self):
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must lie in (0, 1)")
        if not 1 <= self.k <= self.d:
            raise ValueError("k must lie in [1, d]")

    @property
    def epsilon_prime(self) -> float:
        return 2 * self.epsilon / 7

    @property
    def theta(self) -> float:
        return self.epsilon_prime / self.c ** (self.k / 2)

    @property
    def prune_threshold(self) -> float:
        """Features with Inf_i^k[g] >= eps'/k are kept."""
        return self.epsilon_prime / self.k

    @property
    def max_influential(self) -> float:
        """|I| <= 63 k^2 / (8 eps) on the success event."""
        return 63 * self.k ** 2 / (8 * self.epsilon)

    @property
    def ikm(self) -> IKMParameters:
        return IKMParameters(theta=self.theta, delta=self.delta, d=self.d, c=self.c,
                             kappa=self.kappa, max_queries=self.max_queries)

    def as_dict(self) -> dict:
        out = asdict(self)
        out.update(epsilon_prime=self.epsilon_prime, prune_threshold=self.prune_threshold,
                   max_influential=self.max_influential)
        out.update({f"ikm_{key}": value for key, value in self.ikm.as_dict().items()
                    if key not in out})
        return out

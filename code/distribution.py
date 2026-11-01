# -----------------------------------------------------------------------------
#
# Product distributions and their Fourier bases
#
# The instance space is Omega = Omega_1 x ... x Omega_d with Omega_i = {0, ...,
# c_i - 1}, and D = D_1 x ... x D_d has full support. Instances are uint8
# tensors on `device`, which caps every alphabet at 256 values.
#
# One ProductDistribution object is shared by every component that samples from
# D or evaluates a basis function: IKM, the explainers and the evaluator. With a
# single object there is no room for the mismatch of the previous code, where
# the bases were built on a smoothed copy of the marginals while the relevance
# error was estimated on the raw ones.
#
# -----------------------------------------------------------------------------

from typing import Tuple, List, Optional, Sequence

import numpy as np
import torch


class Basis:
    """
    Orthonormal basis of L^2(Omega_i, D_i) with phi_0 = 1, obtained by the
    Stieltjes procedure of the appendix (Algorithm 3), the values 0, ..., c-1
    serving as nodes. phi[j, a] is the value of phi_j at a. Computed once, in
    float64 on the CPU: the basis has c^2 entries.

    No clipping and no smoothing take place here: smoothing, if any, is part of
    the definition of D_i, and clipping would break orthonormality, hence
    Parseval, hence the identity err(K) = (1 - W(f_K)) / 2.
    """

    def __init__(self, distribution: np.ndarray):
        mu = np.asarray(distribution, dtype=float)
        if mu.ndim != 1 or mu.size < 1:
            raise ValueError("A marginal must be a non-empty vector")
        if np.any(mu <= 0) or not np.isclose(mu.sum(), 1.0):
            raise ValueError("A marginal must be a probability vector with full support")

        self.mu = mu
        self.size = mu.size
        self.phi = self._stieltjes(mu)
        self.leakage = float(np.max(np.abs(self.gram() - np.eye(self.size))))

    @staticmethod
    def _stieltjes(mu: np.ndarray) -> np.ndarray:
        c = mu.size
        nodes = np.arange(c, dtype=float)
        phi = np.zeros((c, c))
        phi[0] = 1.0
        phi_prev = np.zeros(c)
        alpha = 0.0
        for j in range(c - 1):
            beta = np.sum(mu * nodes * phi[j] ** 2)
            u = (nodes - beta) * phi[j] - alpha * phi_prev
            alpha = np.sqrt(np.sum(mu * u ** 2))
            phi_prev = phi[j]
            phi[j + 1] = u / alpha
        return phi

    def gram(self) -> np.ndarray:
        """<phi_j, phi_l>_D, which must be the identity."""
        return (self.phi * self.mu) @ self.phi.T


class ProductDistribution:
    def __init__(self, marginals: Sequence[np.ndarray], names: Optional[List[str]] = None,
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.bases = [Basis(m) for m in marginals]
        self.d = len(self.bases)
        self.cards = np.array([b.size for b in self.bases], dtype=int)
        if self.cards.max() > 256:
            raise ValueError("Alphabets are stored as uint8: at most 256 values per feature")
        self.c = int(self.cards.max())
        self.names = list(names) if names is not None else [str(i) for i in range(self.d)]
        self.device = torch.device(device)
        self.dtype = dtype

        # Padded tables: phi[i, j, a] and the cumulative sums of D_i. Setting the
        # cumulative sums to exactly 1 from index c_i - 1 on makes values beyond
        # c_i - 1 unreachable, whatever the rounding of np.cumsum.
        c = self.c
        phi = np.zeros((self.d, c, c))
        cdf = np.ones((self.d, c))
        for i, b in enumerate(self.bases):
            phi[i, :b.size, :b.size] = b.phi
            cdf[i, :b.size - 1] = np.cumsum(b.mu)[:-1]
        self._phi = torch.tensor(phi, dtype=torch.float64, device=self.device)
        self._cdf = torch.tensor(cdf[:, :-1], dtype=torch.float64, device=self.device)
        self._mu = torch.tensor(np.stack([np.pad(b.mu, (0, c - b.size)) for b in self.bases]),
                                dtype=torch.float64, device=self.device)

    # -- Constructors ----------------------------------------------------------

    @classmethod
    def uniform(cls, d: int, c: int, **kwargs) -> "ProductDistribution":
        return cls([np.full(c, 1.0 / c) for _ in range(d)], **kwargs)

    @classmethod
    def from_data(cls, X: np.ndarray, cards: Sequence[int], alpha: float = 1.0,
                  names: Optional[List[str]] = None, **kwargs) -> "ProductDistribution":
        """
        Empirical marginals with Laplace smoothing (counts + alpha), which is
        what guarantees full support when a value is absent from the sample.
        """
        marginals = []
        for i, c_i in enumerate(cards):
            counts = np.bincount(np.asarray(X[:, i], dtype=int), minlength=c_i).astype(float)
            marginals.append((counts + alpha) / (counts.sum() + alpha * c_i))
        return cls(marginals, names=names, **kwargs)

    # -- Sampling --------------------------------------------------------------

    def sample(self, n: int, generator: torch.Generator,
               columns: Optional[Sequence[int]] = None) -> torch.Tensor:
        """n draws from the product of the marginals indexed by `columns` (all by default)."""
        columns = list(range(self.d)) if columns is None else list(columns)
        if not columns:
            return torch.empty((n, 0), dtype=torch.uint8, device=self.device)
        U = torch.rand((n, len(columns)), generator=generator, device=self.device, dtype=torch.float64)
        cdf = self._cdf[columns]                                   # (m, c - 1)
        return (U.unsqueeze(-1) >= cdf.unsqueeze(0)).sum(dim=-1).to(torch.uint8)

    # -- Basis functions -------------------------------------------------------

    def support(self, coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse form of (m, p) multi-indices: features (m, r) and values (m, r),
        with r the largest support size. Shorter supports are padded with the
        value 0, whose basis function phi_{i,0} = 1 leaves the product unchanged.
        """
        coordinates = coordinates.long()
        nonzero = coordinates != 0
        r = max(int(nonzero.sum(dim=1).max()) if coordinates.numel() else 0, 1)
        r = min(r, max(coordinates.shape[1], 1))
        if coordinates.shape[1] == 0:
            zeros = torch.zeros((coordinates.shape[0], 1), dtype=torch.long, device=coordinates.device)
            return zeros, zeros
        features = torch.argsort((~nonzero).to(torch.int8), dim=1, stable=True)[:, :r]
        return features, coordinates.gather(1, features)

    def phi(self, coordinates: torch.Tensor, X: torch.Tensor,
            dtype: Optional[torch.dtype] = None,
            support: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Values of the basis functions phi_s on a batch of (partial) instances.

        coordinates : (m, p) multi-indices on the first p features.
        X           : (L, p) values of the first p features.
        support     : optional precomputed `support(coordinates)`.
        Returns the (m, L) matrix [phi_s(x)], in `dtype` (the distribution's by
        default). Only the nonzero entries of each multi-index are multiplied
        in, since phi_{i,0} = 1: the cost is m * L * r, with r <= k the largest
        support, whatever the prefix length p.
        """
        dtype = dtype or self.dtype
        m = coordinates.shape[0]
        if X.shape[1] == 0:
            return torch.ones((m, X.shape[0]), dtype=dtype, device=self.device)
        features, values = support if support is not None else self.support(coordinates)
        table = self._table(dtype)[: X.shape[1]]         # (p, c, c): [i, s, x]
        p, c = table.shape[0], table.shape[1]
        L = X.shape[0]
        # V[i * c + s, l] = phi_{i,s}(x_{l,i}); then each output row is a product
        # of at most r rows of V, gathered whole.
        V = table.gather(2, X.long().T.unsqueeze(1).expand(p, c, L)).reshape(p * c, L)
        flat = features * c + values
        out = V.index_select(0, flat[:, 0])
        for j in range(1, flat.shape[1]):
            out.mul_(V.index_select(0, flat[:, j]))
        return out

    def _table(self, dtype: torch.dtype) -> torch.Tensor:
        cache = self.__dict__.setdefault("_tables", {})
        if dtype not in cache:
            cache[dtype] = self._phi.to(dtype)
        return cache[dtype]

    def enumerate(self) -> torch.Tensor:
        """All points of Omega, row-major; only sensible for small spaces."""
        grids = torch.meshgrid(*[torch.arange(int(c), device=self.device) for c in self.cards],
                               indexing="ij")
        return torch.stack([g.reshape(-1) for g in grids], dim=1).to(torch.uint8)

    def probabilities(self, X: torch.Tensor) -> torch.Tensor:
        """D(x) for each row of X, in float64."""
        X = X.long()
        out = torch.ones(X.shape[0], dtype=torch.float64, device=self.device)
        for i in range(self.d):
            out *= self._mu[i, X[:, i]]
        return out

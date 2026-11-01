# -----------------------------------------------------------------------------
#
# Membership oracle
#
# The only access the explainers have to f. Every query goes through __call__,
# so that the count reported in the CSV is the query complexity of the
# algorithm, measured the same way for every explainer. The evaluator does not
# go through an oracle: measuring err(K) is not part of explaining.
#
# -----------------------------------------------------------------------------

import torch

from blackbox import BlackBox


class MembershipOracle:
    def __init__(self, blackbox: BlackBox, chunk_size: int = 1 << 20):
        self.blackbox = blackbox
        self.chunk_size = chunk_size
        self.n_queries = 0
        self._checked = False

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Labels in {-1, +1} as int8, one query per row of X."""
        self.n_queries += X.shape[0]
        if X.shape[0] <= self.chunk_size:
            y = self.blackbox.predict(X)
        else:
            y = torch.cat([self.blackbox.predict(X[start:start + self.chunk_size])
                           for start in range(0, X.shape[0], self.chunk_size)])
        if not self._checked:
            self._check(y, X.shape[0])
        return y

    def _check(self, y: torch.Tensor, n: int) -> None:
        """
        Once per oracle: the theory needs f with values in {-1, +1}. A black box
        returning scores, as the GPU path of the previous code did, would make
        every estimate of IKM silently wrong.
        """
        if y.dtype != torch.int8 or y.shape != (n,) or not bool(((y == 1) | (y == -1)).all()):
            raise TypeError(f"{type(self.blackbox).__name__}.predict must return int8 labels in {{-1, +1}}")
        self._checked = True

    def reset(self) -> None:
        self.n_queries = 0

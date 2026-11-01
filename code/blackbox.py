# -----------------------------------------------------------------------------
#
# Black boxes: functions from Omega to {-1, +1}
#
# A trained classifier and a synthetic target are interchangeable: both expose
# predict(X) on uint8 tensors and return int8 labels in {-1, +1} on the same
# device. Synthetic targets are deterministic functions of x, which is what a
# membership oracle requires; the label noise of NoisyParity is therefore a
# fixed pseudo-random function of x rather than a coin tossed at each query.
#
# -----------------------------------------------------------------------------

import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


class BlackBox(ABC):
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Labels in {-1, +1} as an int8 tensor, one per row of X."""


# -- Trained classifiers -------------------------------------------------------

class Classifier(BlackBox):
    """
    A scikit-learn classifier trained on {-1, +1} labels.

    For the MLP, prediction runs on the device of the queries: the forward pass
    is rebuilt from coefs_ and intercepts_, which takes a few lines and
    reproduces MLPClassifier.predict, whose binary rule is logit > 0 (that is,
    probability > 1/2). Other models are queried on the CPU through numpy.
    """

    MODELS = {
        "Decision Tree": DecisionTreeClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "Neural Network": MLPClassifier,
        "Random Forest": RandomForestClassifier,
    }

    ACTIVATIONS = {
        "relu": torch.relu,
        "tanh": torch.tanh,
        "logistic": torch.sigmoid,
        "identity": lambda z: z,
    }

    def __init__(self, name: str = "Neural Network", seed: int = 48, **kwargs):
        if name not in self.MODELS:
            raise ValueError(f"Classifier '{name}' not found. Available: {', '.join(self.MODELS)}")
        self.name = name
        self.seed = seed
        self.model = self.MODELS[name](**kwargs)
        if "random_state" in self.model.get_params() and "random_state" not in kwargs:
            self.model.set_params(random_state=seed)
        self.train_accuracy = float("nan")
        self.test_accuracy = float("nan")
        self._layers = None

    def train(self, X: np.ndarray, Y: np.ndarray, X_test=None, Y_test=None) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(X, Y)
        self.train_accuracy = float(np.mean(self.model.predict(X) == Y))
        if X_test is not None:
            self.test_accuracy = float(np.mean(self.model.predict(X_test) == Y_test))
        self._layers = None
        return self.train_accuracy

    def _torch_layers(self, device: torch.device):
        if self._layers is None or self._layers[0][0].device != device:
            self._layers = [
                (torch.tensor(W, dtype=torch.float32, device=device),
                 torch.tensor(b, dtype=torch.float32, device=device))
                for W, b in zip(self.model.coefs_, self.model.intercepts_)
            ]
        return self._layers

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, MLPClassifier) and len(self.model.classes_) == 2:
            layers = self._torch_layers(X.device)
            act = self.ACTIVATIONS[self.model.activation]
            z = X.to(torch.float32)
            with torch.no_grad():
                for W, b in layers[:-1]:
                    z = act(z @ W + b)
                logit = (z @ layers[-1][0] + layers[-1][1]).flatten()
            classes = torch.tensor(self.model.classes_, dtype=torch.int8, device=X.device)
            return classes[(logit > 0).long()]
        labels = self.model.predict(X.cpu().numpy())
        return torch.as_tensor(labels.astype(np.int8), device=X.device)


# -- Synthetic targets ---------------------------------------------------------

class Parity(BlackBox):
    """
    f(x) = prod_{i in S} (-1)^{x_i}.

    For c = 2 this is the character chi_S. For any even c and uniform marginals,
    each factor (-1)^{x_i} has zero mean, so the whole spectral energy of f lies
    on coordinates with support exactly S: err(K) = 0 if S is included in K and
    1/2 otherwise. No proper subset of S, and no single feature, carries any
    signal, which is precisely what defeats forward selection.
    """

    def __init__(self, support: Sequence[int]):
        self.support = sorted(int(i) for i in support)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if not self.support:
            return torch.ones(X.shape[0], dtype=torch.int8, device=X.device)
        bits = (X[:, self.support] & 1).sum(dim=1) & 1
        return (1 - 2 * bits).to(torch.int8)


class SynergyTrap(Parity):
    """
    The Synergy Trap of the submitted version, with a hidden parity of any size:
    f(x) = +1 if x_g = 1, and chi_S(x) otherwise (g not in S).

    Under the uniform distribution on {0,1}^d, err({}) = 3/8 and err({g}) = 1/4,
    while every subset of S, alone or with g, leaves err unchanged until S is
    complete: forward selection takes the distractor g and then stops, whereas
    K = {g} + S has err = 0 for k = |S| + 1.
    """

    def __init__(self, distractor: int, support: Sequence[int]):
        super().__init__(support)
        if distractor in self.support:
            raise ValueError("The distractor must lie outside the parity")
        self.distractor = int(distractor)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        y = super().predict(X)
        return torch.where(X[:, self.distractor] == 1, torch.ones_like(y), y)


class NoisyParity(Parity):
    """
    Parity whose label is flipped on a fixed pseudo-random set of instances of
    density eta. The set is defined by hashing x (a random linear map followed
    by the splitmix64 finalizer), so that f stays a function. On a space large
    enough for the hash to behave as a random function, the energy of f_S is
    (1 - 2 eta)^2 and opt = 2 eta (1 - eta); for small spaces the brute force
    explainer gives the exact value.

    torch has no uint64 arithmetic: the hash runs on int64, whose products wrap
    around like unsigned ones, with logical shifts emulated by masking and the
    final comparison done on the top 53 bits.
    """

    def __init__(self, support: Sequence[int], eta: float, d: int, seed: int = 0):
        super().__init__(support)
        if not 0 <= eta < 0.5:
            raise ValueError("eta must lie in [0, 1/2)")
        self.eta = eta
        rng = np.random.default_rng(seed)
        weights = rng.integers(0, 2 ** 62, size=d, dtype=np.int64) * 2 + 1  # odd
        self._weights_cpu = torch.tensor(weights, dtype=torch.int64)
        self._weights = {}

    @staticmethod
    def _signed(value: int) -> int:
        return value - (1 << 64) if value >= 1 << 63 else value

    @classmethod
    def _shr(cls, z: torch.Tensor, s: int) -> torch.Tensor:
        return (z >> s) & ((1 << (64 - s)) - 1)

    def _splitmix(self, z: torch.Tensor) -> torch.Tensor:
        z = z + self._signed(0x9E3779B97F4A7C15)
        z = (z ^ self._shr(z, 30)) * self._signed(0xBF58476D1CE4E5B9)
        z = (z ^ self._shr(z, 27)) * self._signed(0x94D049BB133111EB)
        return z ^ self._shr(z, 31)

    def flipped(self, X: torch.Tensor) -> torch.Tensor:
        weights = self._weights.setdefault(X.device, self._weights_cpu.to(X.device))
        acc = ((X.to(torch.int64) + 1) * weights).sum(dim=1)
        top = self._shr(self._splitmix(acc), 11)          # uniform on [0, 2^53)
        return top < int(self.eta * 2 ** 53)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        y = super().predict(X)
        return torch.where(self.flipped(X), -y, y)

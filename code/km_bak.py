import numpy as np
from timeit import default_timer as timer
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
from classifier import Classifier
from utils import Console

@dataclass
class Term:
    """Represents a single term in the Fourier expansion."""
    coefficient: float
    subset: set
    vector: np.ndarray  # Binary indicator vector

class KM:
    def __init__(self, 
                 classifier: Classifier,
                 is_degree_bounded: bool = True,
                 is_sample_bounded: bool = True,
                 confidence: float = 0.95,
                 max_degree: int = 5,
                 max_influence_samples: int = 500,
                 max_training_samples: int = 5000,
                 variable_ordering_heuristic: str = "influence",
                 threshold: float = 0.5, 
                 verbose: bool = True):
        
        self.console = Console(verbose=verbose)
        self.classifier = classifier
        self.dimension = classifier.dimension
        
        # Hypothesis: List of discovered heavy Fourier terms
        self.hypothesis: List[Term] = []

        # Hyperparameters
        self.delta = confidence
        self.theta = threshold
        self.max_influence_samples = max_influence_samples
        self.variable_ordering_heuristic = variable_ordering_heuristic
        self.is_degree_bounded = is_degree_bounded
        self.walltime = 0.0

        # Derived parameters
        self.max_degree = max_degree if is_degree_bounded else self.dimension
        
        # Theoretical complexity: m = O((1/theta^4) * log(d/delta))
        self.sample_complexity = int(
            (32 / (self.theta ** 4)) * np.log(16 * self.dimension / (self.delta * (self.theta ** 2)))
        )
        self.max_training_samples = max_training_samples if is_sample_bounded else self.sample_complexity

    def __str__(self):
        stats = {
            "Dimension (d)": self.dimension,
            "Confidence (delta)": self.delta,
            "Threshold (theta)": self.theta,
            "Max Degree (k)": self.max_degree,
            "Max Samples (m)": self.max_training_samples,
            "Ordering": self.variable_ordering_heuristic,
            "Walltime": f"{self.walltime:.4f}s"
        }
        res = self.console.string("KM Algorithm Information", endl=True)
        for k, v in stats.items():
            res += self.console.string(k, v, endl=True)
        
        res += self.console.string("Discovered Hypothesis", endl=True)
        for term in self.hypothesis:
            subset_str = f"S={term.subset}"
            res += self.console.string(f"  {subset_str.ljust(25)}", f"coef: {term.coefficient:.4f}", endl=True)
        return res

    def fidelity_error(self, X: np.ndarray) -> float:
        r"""
        Estimates the fidelity error P[ f(x) != g(x) ] over the sample set X.
        """

        f_X = self.classifier.predict(X)
        g_X = self.predict(X)
        return np.mean(f_X != g_X)


    def learn(self) -> None:
        self.console.log("Starting KM learning phase...")
        start = timer()
        
        self.hypothesis = []
        self.order = self._get_variable_ordering()
        
        # alpha: -1 is '*', 0/1 are fixed bits
        alpha = np.full(self.dimension, -1, dtype=int)
        self._search(alpha, 0)
        
        self.walltime = timer() - start
        self.console.log(f"Learning complete in {self.walltime:.2f}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized prediction: $g(x) = \text{sgn}(\sum \hat{f}(s) (-1)^{s \cdot x})$
        """
        if not self.hypothesis:
            return np.ones(X.shape[0]) if X.ndim > 1 else 1.0

        # Collect all coefficients and indicator vectors
        coeffs = np.array([t.coefficient for t in self.hypothesis])
        vectors = np.stack([t.vector for t in self.hypothesis]) # shape: (|H|, d)

        # Vectorized parity calculation for all samples and all terms
        # X: (N, d), vectors.T: (d, |H|) -> dot_products: (N, |H|)
        dot_products = X @ vectors.T
        parities = (-1) ** dot_products
        
        # Weighted sum across terms
        expansion = parities @ coeffs
        
        preds = np.where(expansion >= 0, 1, -1)
        return preds if X.ndim > 1 else preds[0]

    def _search(self, alpha: np.ndarray, depth: int) -> None:
        """Recursive backtracking search with pruning."""
        weight = self._estimate_weight(alpha)
        
        # Pruning: If the total weight of all extensions is small, skip this branch
        if weight < (self.theta ** 2) / 2:
            return

        # Base Case: Leaf node
        if depth == self.dimension:
            coef = self._estimate_coefficient(alpha)
            if abs(coef) >= (self.theta / 2): # Secondary check for stability
                self.hypothesis.append(Term(
                    coefficient=coef, 
                    subset=self._get_support(alpha), 
                    vector=alpha.copy()
                ))
            return

        # Branching
        idx = self.order[depth]
        current_degree = np.sum(alpha == 1)

        # Branch 0 (idx is not in subset)
        alpha[idx] = 0
        self._search(alpha, depth + 1)
        
        # Branch 1 (idx is in subset) - check degree constraint
        if not self.is_degree_bounded or current_degree < self.max_degree:
            alpha[idx] = 1
            self._search(alpha, depth + 1)
        
        # Backtrack
        alpha[idx] = -1

    def _estimate_coefficient(self, s_vec: np.ndarray, m: Optional[int] = None) -> float:
        """
        Estimates the Fourier coefficient $\hat{f}(s) = \mathbb{E}_{x}[f(x)\chi_s(x)]$.
        """
        m = m or self.max_training_samples
        Z = np.random.randint(2, size=(m, self.dimension))
        f_Z = self.classifier.predict(Z)
        
        # Parity function chi_s(x) = (-1)^{s \cdot x}
        # Only indices where s_vec == 1 contribute to the dot product
        parity = (-1) ** (Z @ s_vec)
        return np.mean(parity * f_Z)

    def _estimate_weight(self, alpha: np.ndarray) -> float:
        """
        Estimates the weight of the Fourier transform 'above' a prefix using the 
        Parseval-based conditional expectation trick.
        """
        d, m = self.dimension, self.max_training_samples
        fixed_mask = (alpha != -1)
        
        # Generate samples with shared suffixes but independent bits for fixed prefix indices
        Suffixes = np.random.randint(2, size=(m, d))
        Z1, Z2 = Suffixes.copy(), Suffixes.copy()
        
        Z1[:, fixed_mask] = np.random.randint(2, size=(m, np.sum(fixed_mask)))
        Z2[:, fixed_mask] = np.random.randint(2, size=(m, np.sum(fixed_mask)))

        f_z1 = self.classifier.predict(Z1) 
        f_z2 = self.classifier.predict(Z2)

        phi_z1 = (-1) ** (Z1[:, fixed_mask] @ alpha[fixed_mask])
        phi_z2 = (-1) ** (Z2[:, fixed_mask] @ alpha[fixed_mask])

        return np.mean(phi_z1 * f_z1 * phi_z2 * f_z2)

    def _get_variable_ordering(self) -> np.ndarray:
        """Determines search order based on influence or randomness."""
        if self.variable_ordering_heuristic == "influence":
            self.console.log("Ranking variables by influence...")
            influences = []
            for i in range(self.dimension):
                s = np.zeros(self.dimension, dtype=int)
                s[i] = 1
                # Influence of variable i is roughly the magnitude of its singleton coefficient
                c = self._estimate_coefficient(s, m=self.max_influence_samples)
                influences.append(abs(c))
            return np.argsort(influences)[::-1]
        
        elif self.variable_ordering_heuristic == "random":
            return np.random.permutation(self.dimension)
            
        return np.arange(self.dimension)

    def _get_support(self, s: np.ndarray) -> set:
        return {int(i) for i in np.where(s == 1)[0]}
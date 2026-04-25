import numpy as np
import itertools
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer
from typing import List, Tuple, Union, Optional, Set
from dataclasses import dataclass
from classifier import Classifier
from utils import Console

@dataclass
class Term:
    coefficient: float
    subset: set
    vector: np.ndarray

class KM:
    def __init__(self, 
                 classifier: Classifier,
                 is_degree_bounded: bool = True,
                 is_sample_bounded: bool = True,
                 confidence: float = 0.95,
                 max_degree: int = 5,
                 max_training_samples: int = 5000,
                 variable_ordering_heuristic: str = "influence",
                 threshold: float = 0.5, 
                 split_depth: int = 4,  # How many variables to fix for parallelization
                 verbose: bool = True):
        
        self.console = Console(verbose=verbose)
        self.classifier = classifier
        self.dimension = classifier.dimension
        self.hypothesis: List[Term] = []

        # Hyperparameters
        self.theta = threshold
        self.delta = confidence
        self.max_degree = max_degree
        self.is_degree_bounded = is_degree_bounded
        self.max_training_samples = max_training_samples
        self.variable_ordering_heuristic = variable_ordering_heuristic
        self.split_depth = min(split_depth, self.dimension)
        self.walltime = 0.0

        # Multi-processing setup
        cpu_cores = os.cpu_count() or 1
        self.n_workers = min(cpu_cores, 61) if platform.system() == "Windows" else cpu_cores
        
    def __str__(self):
        stats = {
            "Dimension (d)": self.dimension,
            "Threshold (theta)": self.theta,
            "Max Degree (k)": self.max_degree,
            "Max Samples (m)": self.max_training_samples,
            "Workers": self.n_workers,
            "Split Depth": self.split_depth,
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
        self.console.log(f"Starting Parallel KM (d={self.dimension}, workers={self.n_workers})...")
        start = timer()
        
        # 1. Determine variable ordering (must be done sequentially)
        self.order = self._get_variable_ordering()
        
        # 2. Generate prefixes for parallel tasks
        # We fix the first 'split_depth' variables in the ordering
        prefixes = []
        # Generate all 2^split_depth combinations of {0, 1}
        for bits in itertools.product([0, 1], repeat=self.split_depth):
            alpha = np.full(self.dimension, -1, dtype=int)
            for i, bit in enumerate(bits):
                idx = self.order[i]
                alpha[idx] = bit
            prefixes.append(alpha)

        # 3. Dispatch to ProcessPoolExecutor
        self.hypothesis = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for alpha_prefix in prefixes:
                futures.append(executor.submit(
                    KM._search_worker,
                    alpha_prefix,
                    self.split_depth,
                    self.dimension,
                    self.order,
                    self.classifier,
                    self.theta,
                    self.max_degree,
                    self.is_degree_bounded,
                    self.max_training_samples
                ))
            
            for future in futures:
                local_hypo = future.result()
                self.hypothesis.extend(local_hypo)

        self.walltime = timer() - start
        self.console.log(f"Parallel learning complete in {self.walltime:.2f}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.hypothesis:
            return np.ones(X.shape[0]) if X.ndim > 1 else 1.0
        coeffs = np.array([t.coefficient for t in self.hypothesis])
        vectors = np.stack([t.vector for t in self.hypothesis])
        dot_products = X @ vectors.T
        parities = (-1) ** dot_products
        expansion = parities @ coeffs
        return np.where(expansion >= 0, 1, -1)

    @staticmethod
    def _search_worker(alpha: np.ndarray, 
                       depth: int, 
                       dimension: int,
                       order: np.ndarray,
                       classifier: Classifier,
                       theta: float,
                       max_degree: int,
                       is_degree_bounded: bool,
                       m: int) -> List[Term]:
        """Independent search tree worker."""
        local_hypothesis = []
        
        def _estimate_weight(a: np.ndarray) -> float:
            fixed_mask = (a != -1)
            Suffixes = np.random.randint(2, size=(m, dimension))
            Z1, Z2 = Suffixes.copy(), Suffixes.copy()
            Z1[:, fixed_mask] = np.random.randint(2, size=(m, np.sum(fixed_mask)))
            Z2[:, fixed_mask] = np.random.randint(2, size=(m, np.sum(fixed_mask)))
            f_z1, f_z2 = classifier.predict(Z1), classifier.predict(Z2)
            phi_z1 = (-1) ** (Z1[:, fixed_mask] @ a[fixed_mask])
            phi_z2 = (-1) ** (Z2[:, fixed_mask] @ a[fixed_mask])
            return np.mean(phi_z1 * f_z1 * phi_z2 * f_z2)

        def _estimate_coefficient(s_vec: np.ndarray) -> float:
            Z = np.random.randint(2, size=(m, dimension))
            f_Z = classifier.predict(Z)
            parity = (-1) ** (Z @ s_vec)
            return np.mean(parity * f_Z)

        def _recursive_search(curr_alpha: np.ndarray, curr_depth: int):
            weight = _estimate_weight(curr_alpha)
            if weight < (theta ** 2) / 2:
                return

            if curr_depth == dimension:
                coef = _estimate_coefficient(curr_alpha)
                if abs(coef) >= (theta / 2):
                    support = {int(i) for i in np.where(curr_alpha == 1)[0]}
                    local_hypothesis.append(Term(coef, support, curr_alpha.copy()))
                return

            idx = order[curr_depth]
            current_degree = np.sum(curr_alpha == 1)

            # Branch 0
            curr_alpha[idx] = 0
            _recursive_search(curr_alpha, curr_depth + 1)
            
            # Branch 1
            if not is_degree_bounded or current_degree < max_degree:
                curr_alpha[idx] = 1
                _recursive_search(curr_alpha, curr_depth + 1)
            
            curr_alpha[idx] = -1

        # Start recursion from the provided prefix depth
        _recursive_search(alpha, depth)
        return local_hypothesis

    def _get_variable_ordering(self) -> np.ndarray:
        """Sequential variable ranking by influence."""
        if self.variable_ordering_heuristic == "influence":
            self.console.log("Ranking variables by influence...")
            influences = []
            m = 500 # Smaller sample size for ordering to save time
            for i in range(self.dimension):
                s = np.zeros(self.dimension, dtype=int)
                s[i] = 1
                Z = np.random.randint(2, size=(m, self.dimension))
                f_Z = self.classifier.predict(Z)
                c = np.mean(((-1) ** (Z @ s)) * f_Z)
                influences.append(abs(c))
            return np.argsort(influences)[::-1]
        return np.arange(self.dimension)



    
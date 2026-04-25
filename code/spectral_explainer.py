import math
import itertools
from classifier import Classifier
from km import Term 
import numpy as np
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer
from typing import Set, List, Tuple, Dict
from dataclasses import dataclass
from utils import Console


class SpectralExplainer:
    def __init__(self,
                 classifier: Classifier, 
                 hypothesis: List[Term],
                 is_bounded: bool = True,
                 is_pruned: bool = True,
                 max_size: int = 5, 
                 max_iterations: int = 10_000_000,
                 threshold: float = 0.01, 
                 verbose: bool = True):
        
        self.console = Console(verbose=verbose)
        self.classifier = classifier
        self.dimension = classifier.dimension
        self.hypothesis = hypothesis
        self.is_bounded = is_bounded
        self.is_pruned = is_pruned
        self.max_size = max_size
        self.max_iterations = max_iterations
        self.threshold = threshold
        
        # Results
        self.explanation: Set[int] = set()
        self.energy = 0.0
        self.is_timed_out = False
        self.walltime = 0.0
        
        # 1. Identify raw support (all features mentioned in the hypothesis)
        all_features = set().union(*[term.subset for term in self.hypothesis])
        self.support = sorted(list(all_features))
        
        # 2. GKK Pruning (Optional)
        # Filters features whose total influence (sum of squares of coefficients) is below threshold
        if self.is_pruned and self.hypothesis:
            influences = {f: 0.0 for f in self.support}
            for term in self.hypothesis:
                weight = term.coefficient ** 2
                for f in term.subset:
                    influences[f] += weight
            
            self.support = [f for f in self.support if influences[f] >= self.threshold]
            self.console.log(f"GKK Pruning: {len(all_features)} -> {len(self.support)} features")

        # 3. Bitmask Preparation
        # Mapping global feature indices to local bit positions (0 to len(support)-1)
        self.feat_to_bit = {f: i for i, f in enumerate(self.support)}
        self.subset_masks = []
        self.squares = []
        
        for term in self.hypothesis:
            # Only include terms whose variables survived pruning
            if all(f in self.feat_to_bit for f in term.subset):
                mask = 0
                for f in term.subset:
                    mask |= (1 << self.feat_to_bit[f])
                self.subset_masks.append(mask)
                self.squares.append(term.coefficient ** 2)

        # 4. Multi-processing setup
        cpu_cores = os.cpu_count() or 1
        self.n_workers = min(cpu_cores, 61) if platform.system() == "Windows" else cpu_cores

    def __str__(self):
        output = self.console.string("Spectral Explainer Results", endl=True)
        walltime_display = "TO" if self.is_timed_out else f"{self.walltime:.4f}s"        
        stats = {
            "Active Features": len(self.support),
            "Bounded Search": self.is_bounded,
            "Max Size (k)": self.max_size,
            "Workers": self.n_workers,
            "Explanation K": self.explanation,
            "Energy W(g_K)": f"{self.energy:.4f}",
            "Walltime": walltime_display
        }
        for k, v in stats.items():
            output += self.console.string(k, v, endl=True)
        return output

    def explain(self) -> Tuple[Set[int], float]:
        r"""
        Main method to find the best explanation K that maximizes energy W(g_K).
        If is_bounded is False, this simply returns the full support as the explanation.
        If is_bounded is True, this searches over all subsets of the support of size <= max_size.
        If the number of combinations exceeds max_iterations, it falls back to the unbounded explanation and sets a timeout flag.
        """
        self.is_timed_out = False
        if not self.is_bounded or len(self.support) <= self.max_size:
            return self._find_unbounded_explanation()
        
        total_combos = math.comb(len(self.support), self.max_size)        
        if total_combos > self.max_iterations:
            self.console.log(f"Complexity Error", f"{total_combos} combos exceeds max_iterations")
            self.is_timed_out = True
            return self._find_unbounded_explanation()        
        
        return self._find_bounded_explanation()
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        r"""
        Returns the output X_S = {x_S : x in X} where S is the current explanation.
        For example, if S = {1,3} and X is a matrix of shape (m, d), then this returns the submatrix of shape (m, 2) containing only columns 1 and 3.
        When S = {}, this returns the all-zero vector of shape (m, 0). 
        """
        if self.explanation == set():
            return np.zeros((X.shape[0], 0), dtype=X.dtype)
        indices = sorted(list(self.explanation))
        return X[:, indices]
            
    def relevance_error(self, X: np.ndarray, Z: np.ndarray) -> float:
        r"""
        Estimates Relevance Error: P[ f(x) != f(z) | x_K = z_K ] 
        using samples (x, z) from the uniform distribution.
        """
        # 1. Get classifier predictions for both sets
        f_X = self.classifier.predict(X)
        f_Z = self.classifier.predict(Z)

        # 2. Handle the empty explanation case (K = ∅)
        # If K is empty, the condition x_K = z_K is always true for all pairs.
        # The error is simply the probability that two random inputs have different labels.
        if not self.explanation:
            return np.mean(f_X != f_Z)

        # 3. Identify pairs where the condition x_K = z_K holds
        X_K = self.predict(X)
        Z_K = self.predict(Z)
    
        # Check for equality across all selected feature columns
        match_mask = np.all(X_K == Z_K, axis=1)
    
        # 4. Calculate conditional expectation
        # If no pairs match in this batch, we return 0.0 or NaN to avoid division by zero.
        if not np.any(match_mask):
            self.console.log("Warning: No samples satisfied x_K = z_K. Increase sample size.")
            return 0.0

        return np.mean(f_X[match_mask] != f_Z[match_mask])

    def _find_unbounded_explanation(self) -> Tuple[Set[int], float]:
        self.explanation = set(self.support)
        self.energy = sum(self.squares)
        return self.explanation, self.energy

    
    def _find_bounded_explanation(self) -> Tuple[Set[int], float]:
        n = len(self.support)
        k = self.max_size
        total_combos = math.comb(n, k)
        
        self.console.log(f"Analyzing combinations (C({n}, {k}))", f"{total_combos}")
        start_time = timer()

        def mask_generator():
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for bit in combo:
                    mask |= (1 << bit)
                yield mask

        # Parallel Execution
        best_energy = -1.0
        best_mask = 0
        chunk_size = max(1000, total_combos // (self.n_workers * 2))
        gen = mask_generator()

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            while True:
                chunk = list(itertools.islice(gen, chunk_size))
                if not chunk:
                    break
                futures.append(executor.submit(
                    SpectralExplainer._optimize_chunk, chunk, self.subset_masks, self.squares
                ))
            
            for future in futures:
                res_mask, res_energy = future.result()
                if res_energy > best_energy:
                    best_energy = res_energy
                    best_mask = res_mask

        self.explanation = {self.support[i] for i in range(n) if (best_mask & (1 << i))}
        self.energy = best_energy
        self.walltime = timer() - start_time
        
        return self.explanation, self.energy

    @staticmethod
    def _optimize_chunk(combos_chunk: List[int], subset_masks: List[int], squares: List[float]) -> Tuple[int, float]:
        """
        Calculates energy for a chunk of candidate explanations.
        Energy W(g_K) = sum of squares of coefficients whose subsets are entirely within K.
        """
        best_energy = -1.0
        best_K_mask = 0
        
        for K_mask in combos_chunk:
            current_energy = 0.0
            for s_mask, c_sq in zip(subset_masks, squares):
                if (s_mask & K_mask) == s_mask:
                    current_energy += c_sq            
            if current_energy > best_energy:
                best_energy = current_energy
                best_K_mask = K_mask
        return best_K_mask, best_energy
    
    

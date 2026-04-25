import math
import itertools
import multiprocessing
import numpy as np
import os
import platform

from classifier import Classifier
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer
from typing import Set, List, Tuple
from utils import Console

class KM_Explainer:
    def __init__(self, 
                 classifier: Classifier, 
                 hypothesis: Tuple[List[set], List[float]],
                 is_size_bounded: bool = True,
                 max_size: int = 5, 
                 max_iterations: int = 10_000_000, 
                 max_test_samples: int = 5000, 
                 verbose = True):
        self.console = Console(verbose=verbose)
        self.classifier = classifier
        self.subsets, self.coefficients = hypothesis
        self.explanation = set()
        self.energy = 0.0
        self.is_size_bounded = is_size_bounded
        self.max_iterations = max_iterations
        self.max_size = max_size
        self.max_test_samples = max_test_samples
        
        # 1. Map involved features to bit positions
        self.involved_features = sorted(list(set().union(*self.subsets)))
        self.feat_to_bit = {f: i for i, f in enumerate(self.involved_features)}
        
        # 2. Pre-calculate bitmasks for subsets
        self.subset_masks = []
        for s in self.subsets:
            mask = 0
            for f in s:
                mask |= (1 << self.feat_to_bit[f])
            self.subset_masks.append(mask)
           
        # 3. Platform-safe worker count
        cpu_cores = os.cpu_count() or 1
        if platform.system() == "Windows":
            self.n_workers = min(cpu_cores, 61)
        else:
            self.n_workers = min(cpu_cores, 64)
            
        # 4. Energy pre-calculations (Squares)
        self.squares = [c**2 for c in self.coefficients]
        self.walltime = 0.0

    def __str__(self):
        output = self.console.string("KM Explainer information", endl = True)
        output += self.console.string("Number of features (involved)", len(self.involved_features), endl = True)
        output += self.console.string("Is size bounded?", self.is_size_bounded, endl = True)
        output += self.console.string("Max size (k)", self.max_size, endl = True)
        output += self.console.string("Number of workers", self.n_workers, endl = True)
        output += self.console.string("Wall time (seconds)", f"{self.walltime:.4f}", endl = True)
        output += self.console.string("Explanation (K)", str(self.explanation), endl = True)
        output += self.console.string("Energy W(g_K)", f"{self.energy:.4f}", endl = True)
        return output

    @staticmethod
    def _optimize(combos_chunk: list, subset_masks: list, squares: list) -> Tuple[int, float]:
        """ Static compute engine for energy maximization via bitmasks. """
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
    
    def explain(self) -> Tuple[Set[int], float]:
        """ Entry point for generating the explanation. """
        if self.is_size_bounded:
            return self.find_bounded_explanation()
        else:
            return self.find_unbounded_explanation()

    def find_unbounded_explanation(self) -> Tuple[Set[int], float]:
        """ Returns the full support of hypothesis g. """
        K_support = set(self.involved_features)
        K_energy = sum(self.squares)
        self.explanation = K_support
        self.energy = K_energy
        self.walltime = 0.0  
        return K_support, K_energy
        
    def find_bounded_explanation(self) -> Tuple[Set[int], float]:
        """ Finds the best subset K of size k via parallel exhaustive search. """
        n = len(self.involved_features)
        k = min(self.max_size, n)
        total_combos = math.comb(n, k)
        
        if total_combos > self.max_iterations:
            raise ValueError(f"Combinatorial search space too large: {total_combos} combinations.")

        self.console.log(f"Searching for optimal K with size {k} (Combos: {total_combos})...")
        start_time = timer()

        def mask_generator():
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for bit in combo:
                    mask |= (1 << bit)
                yield mask

        # --- Fast Path for small combination counts ---
        if total_combos < 50_000:
            best_mask, best_energy = self._optimize(list(mask_generator()), self.subset_masks, self.squares)
        else:
            # --- Parallel Path for large combination counts ---
            best_energy = -1.0
            best_mask = 0
            chunk_size = 100_000
            gen = mask_generator()

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                while True:
                    chunk = list(itertools.islice(gen, chunk_size))
                    if not chunk: break
                    futures.append(executor.submit(
                        KM_Explainer._optimize, chunk, self.subset_masks, self.squares
                    ))
                
                for future in futures:
                    res_mask, res_energy = future.result()
                    if res_energy > best_energy:
                        best_energy = res_energy
                        best_mask = res_mask

        # Map bitmask back to original feature indices
        self.explanation = {self.involved_features[i] for i in range(n) if (best_mask & (1 << i))}
        self.energy = best_energy
        self.walltime = timer() - start_time
        
        return self.explanation, self.energy
    
    def estimate_relevance_error(self, K: Set[int]) -> float:
        """ Estimates P[f(x) != f(z) | x_K = z_K] using vectorized NumPy operations. """
        m, d = self.max_test_samples, self.classifier.dimension
        
        # 1. Sample two independent sets
        X = np.random.randint(2, size=(m, d))
        Z = np.random.randint(2, size=(m, d))
        
        # 2. Vectorized glueing: Z[K] = X[K]
        if K:
            indices = list(K)
            Z[:, indices] = X[:, indices]
            
        # 3. Predict and compare
        f_X = self.classifier.predict(X)
        f_Z = self.classifier.predict(Z)

        relevance_error = np.mean(f_X != f_Z)

        if self.console.verbose:
            self.console.log("Empirical Relevance Error", f"{relevance_error:.4f}", endl=True)
        
        return relevance_error
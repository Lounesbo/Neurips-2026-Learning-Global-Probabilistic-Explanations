def gpe_extract(coordinates, coefficients, k, epsilon):
    # Step 2: Feature Pruning
    # Calculate R_i^k for each feature
    d = len(coordinates[0]) if coordinates else 0
    influences = np.zeros(d)
    
    for s_vec, coeff in zip(coordinates, coefficients):
        if np.sum(s_vec) <= k: # Only consider subsets up to size k
            weight = coeff**2
            influences[np.where(s_vec == 1)] += weight
            
    # I = {i : R_i^k[g] >= epsilon/k}
    I = [i for i, inf in enumerate(influences) if inf >= epsilon/k]
    
    # Step 3: Subset Selection
    # Find K subset of I, |K| <= k, maximizing sum(coeff^2 where S subset K)
    best_K = []
    max_energy = -1
    
    from itertools import combinations
    for K_candidate in combinations(I, k):
        current_energy = 0
        K_set = set(K_candidate)
        
        for s_vec, coeff in zip(coordinates, coefficients):
            # Check if subset S is contained in K_candidate
            s_indices = set(np.where(s_vec == 1)[0])
            if s_indices.issubset(K_set):
                current_energy += coeff**2
                
        if current_energy > max_energy:
            max_energy = current_energy
            best_K = K_candidate
            
    return best_K, max_energy



def _sample_manifold(self, m: int) -> np.ndarray:
        """
        Internal helper to sample m instances that strictly respect 
        the 'exactly-one' one-hot categorical constraint.
        """
        d = self.classifier.dimension
        X = np.zeros((m, d), dtype=int)
        rows = np.arange(m)
        
        for group in self.groups:
            # Randomly pick one index from the current group for every row
            choices = np.random.choice(group, size=m)
            # Vectorized assignment: set the chosen bit to 1
            X[rows, choices] = 1
        return X

    def estimate_relevance_error(self, K: Set[int]) -> float:
        r"""
        Estimates the Relevance Error while constrained to the legal data manifold.
        P[f(x) != f(z) | x_K = z_K, x,z \in Manifold]
        """
        m = self.max_test_samples
        
        # 1. Generate a legal base set X
        X = self._sample_manifold(m)
        
        # 2. Construct Z by either copying X or sampling a new valid state
        Z = np.zeros_like(X)
        rows = np.arange(m)
        
        for group in self.groups:
            # Logical condition: Does the explanation K care about this feature group?
            if any(idx in K for idx in group):
                # If YES: Z must match X exactly to satisfy x_K = z_K
                Z[:, group] = X[:, group]
            else:
                # If NO: Z samples a fresh, valid one-hot state for this group
                choices = np.random.choice(group, size=m)
                Z[rows, choices] = 1
                    
        # 3. Compute model agreement
        f_X = self.classifier.predict(X)
        f_Z = self.classifier.predict(Z)
        
        # Disagreement rate
        relevance_error = np.mean(f_X != f_Z)
        
        if self.console.verbose:
            self.console.log("Manifold-Aware Relevance Error", f"{relevance_error:.4f}")
            
        return relevance_error

def sample_manifold(self, m: int) -> np.ndarray:
        """ Samples m instances that respect the 'exactly-one' one-hot constraint. """
        d = self.classifier.dimension
        X = np.zeros((m, d), dtype=int)
        
        # We assume self.groups is a list of index groups: [[0,1,2], [3,4,5,6], ...]
        for group in self.groups:
            # For each instance, pick one index in the group to set to 1
            choices = np.random.choice(group, size=m)
            for row_idx, col_idx in enumerate(choices):
                X[row_idx, col_idx] = 1
        return X

    def estimate_relevance_error(self, K: Set[int]) -> float:
        """ Manifold-aware Relevance Error estimation. """
        m = self.max_test_samples
        X = self.sample_manifold(m)
        Z = np.zeros_like(X)
        
        # 2. Construct Z based on the categorical groups
        for group in self.groups:
            # Check if any bit of this categorical feature is in our explanation K
            if any(idx in K for idx in group):
                # If the feature is 'fixed' by K, Z must match X exactly on this group
                Z[:, group] = X[:, group]
            else:
                # If not fixed, Z samples a fresh legal state for this feature
                choices = np.random.choice(group, size=m)
                for row_idx, col_idx in enumerate(choices):
                    Z[row_idx, col_idx] = 1
                    
        f_X = self.classifier.predict(X)
        f_Z = self.classifier.predict(Z)
        
        relevance_error = np.mean(f_X != f_Z)
        if self.console.verbose:
            self.console.log("Manifold-Aware Relevance Error", f"{relevance_error:.4f}")
        return relevance_error

def estimate_fidelity_error(self, m: int = 5000) -> float:
        """ Measures how well g mimics f on legal instances only. """
        # 1. Sample legal instances
        X = self.sample_manifold(m) 
        
        # 2. Get predictions from the black-box f
        f_X = self.classifier.predict(X)
        
        # 3. Get predictions from our Fourier expansion g
        # Recall: g(x) = sgn(sum c_s * chi_s(x))
        g_X = self.predict(X) # Assuming KM class has a predict method
        
        fidelity_error = np.mean(f_X != g_X)
        return fidelity_error
    
import itertools
import multiprocessing
import numpy as np  
from classifier import Classifier
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer
from typing import Set, List, Tuple
from utils import Console


# We need a top-level function or a static method for multiprocessing to serialize it
def _worker_search(features_chunk, subsets, coefficients, k):
    best_energy = -1.0
    best_K = set()
    for c in features_chunk:
        current_K = set(c)
        # Inline the energy calculation for speed
        energy = 0.0
        for s, coef in zip(subsets, coefficients):
            if s.issubset(current_K):
                energy += coef**2
        if energy > best_energy:
            best_energy = energy
            best_K = current_K
    return best_K, best_energy

# Inside your KM_Explainer class:
def explain_parallel(self, n_workers=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
        
    n = len(self.features)
    k = min(self.max_size, n)
    start_time = timer()

    # Generate all combinations
    all_combos = list(itertools.combinations(self.features, k))
    
    # Split combinations into chunks for each worker
    chunk_size = len(all_combos) // n_workers
    chunks = [all_combos[i:i + chunk_size] for i in range(0, len(all_combos), chunk_size)]

    best_overall_K = set()
    max_overall_energy = -1.0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Pass the constants (subsets, coefficients) to every worker
        futures = [
            executor.submit(_worker_search, chunk, self.subsets, self.coefficients, k) 
            for chunk in chunks
        ]
        
        for future in futures:
            res_K, res_energy = future.result()
            if res_energy > max_overall_energy:
                max_overall_energy = res_energy
                best_overall_K = res_K

    self.walltime = timer() - start_time
    return best_overall_K, max_overall_energy

class KM_Explainer:
    def __init__(self, 
                 classifier: Classifier, 
                 hypothesis: Tuple[List[set], List[float]], 
                 max_size: int, 
                 max_test_samples: int = 5000, 
                 verbose = True):
        r"""
        Initializes the KM Explainer with the KM hypothesis g and size limit k.
        """
        self.console = Console(verbose=verbose)
        self.classifier = classifier
        self.subsets, self.coefficients = hypothesis
        self.features = list(set().union(*self.subsets))
        self.max_size = max_size
        self.max_test_samples = max_test_samples
        self.walltime = 0.0
        
    def __str__(self):
        r"""
        String representation of the KM Explainer information.
        """
        str = self.console.string("KM Explainer information", endl = True)
        str += self.console.string("Number of features", len(self.features), endl = True)
        str += self.console.string("Maximum size", self.max_size, endl = True)
        str += self.console.string("Wall time (seconds)", self.walltime, endl = True)
        return str
        
    def get_energy(self, K: Set[int]) -> float:
        r"""
        Calculates the energy W(g_K) for a given subset of features K.
        """
        energy = 0.0
        for s, c in zip(self.subsets, self.coefficients):
            # A coefficient is captured only if its support is a subset of K
            if s.issubset(K):
                energy += c**2
        return energy
    
    def estimate_relevance_error(self, K: Set[int]) -> float:
        r"""
        Estimates the relevance error: P_{(x,z) ~ U x U} [ f(x) != f(z) | x_K = z_K]
        """

        m = self.max_test_samples
        d = self.classifier.dimension
        
        # 1. Sample x uniformly from the hypercube
        X = np.random.randint(2, size=(m, d))
        
        # 2. Sample z such that it matches x on K, but is random elsewhere
        # We start with a random hypercube sample...
        Z = np.random.randint(2, size=(m, d))
        # ...and "glue" the features from K to match X
        for feature_idx in K:
            Z[:, feature_idx] = X[:, feature_idx]
            
        # 3. Query the base classifier f (the Neural Net)
        f_X = self.classifier.predict(X)
        f_Z = self.classifier.predict(Z)
        
        # 4. The error is the probability they disagree
        # (Since labels are in {-1, 1}, f(x) != f(z) is the logic)
        relevance_error = np.mean(f_X != f_Z)
        
        if self.console.verbose:
            self.console.log("Estimated Relevance Error", relevance_error)
            
        return relevance_error
    
    def explain(self) -> Tuple[Set[int], float]:
        r"""
        Finds the subset K of size k that maximizes energy of g_K.
        """
        n = len(self.features)
        k = min(self.max_size, n)  # Ensure k does not exceed number of features
        start_time = timer()
        best_K = set()
        max_energy = -1.0
                
        for c in itertools.combinations(self.features, k):
            current_K = set(c)
            energy = self.get_energy(current_K)
            if energy > max_energy:
                max_energy = energy
                best_K = current_K
                
        end_time = timer()
        self.walltime = end_time - start_time

        return best_K, max_energy
        

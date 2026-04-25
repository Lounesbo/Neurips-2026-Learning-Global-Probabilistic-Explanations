import numpy as np
import pandas as pd
from typing import Set, List, Tuple, Union
from km import KM
from classifier import Classifier
from utils import Console

class Sampler:
    def __init__(self,
                 dimension: int,
                 distribution: str = "uniform", 
                 groups: List[List[int]] = [],
                 max_samples: int = 5000,
                 verbose: bool = True):        

        self.console = Console(verbose=verbose)
        self.dimension = dimension
        self.distribution = distribution
        self.max_samples = max_samples
        
        # Define groups based on distribution
        if distribution == "uniform":
            # Each feature is independent
            self.groups = [[i] for i in range(dimension)] 
        elif distribution == "exactly-one":
            # Expects a list of lists, e.g., [[0,1,2], [3,4]] for one-hot groups
            self.groups = groups
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        self.nb_groups = len(self.groups)
        
    def __str__(self):
        res = self.console.string("Sampler Information", endl=True)
        res += self.console.string("Distribution", self.distribution, endl=True)        
        res += self.console.string("Dimension", self.dimension, endl=True)
        res += self.console.string("Max Samples", self.max_samples, endl=True)
        return res

    def sample_manifold(self, m: int) -> np.ndarray:
        """ 
        Samples m instances that respect the distributional constraints. 
        """
        if self.distribution == "uniform":
            return np.random.randint(2, size=(m, self.dimension))
        
        # Exactly-one manifold (One-Hot constraints)
        X = np.zeros((m, self.dimension), dtype=int)
        rows = np.arange(m)        
        for group in self.groups:
            choices = np.random.choice(group, size=m)
            X[rows, choices] = 1
        return X

    def estimate_fidelity_error(self, f: Classifier, g: KM) -> float:
        """
        Estimates P[ f(x) != g(x) ] over the manifold.
        """
        X = self.sample_manifold(self.max_samples) 
        
        f_X = f.predict(X)
        g_X = g.predict(X)
        
        error = np.mean(f_X != g_X)
        if self.console.verbose:
            self.console.log("Fidelity Error", f"{error:.4f}")
        return error

    def estimate_relevance_error(self, f: Classifier, K: Set[int]) -> float:
        """
        Estimates Relevance Error: P[ f(x) != f(z) | x_K = z_K ]
        while strictly remaining on the legal data manifold.
        """
        m = self.max_samples
        X = self.sample_manifold(m)
        Z = np.zeros_like(X)
        rows = np.arange(m)
        
        for group in self.groups:
            # Check if any feature in this constraint group is part of the explanation K
            is_group_fixed = any(idx in K for idx in group)
            
            if is_group_fixed:
                # To satisfy x_K = z_K on a one-hot manifold, we MUST copy the 
                # entire group. If we changed a non-K bit in this group, 
                # the one-hot constraint would force the K-bit to zero.
                Z[:, group] = X[:, group]
            else:
                # Randomize this group independently
                if self.distribution == "uniform":
                    # group is a single index, randomize 0/1
                    Z[:, group] = np.random.randint(2, size=(m, 1))
                else:
                    # sample a new valid one-hot state for the categorical group
                    choices = np.random.choice(group, size=m)
                    Z[rows, choices] = 1
                    
        f_X = f.predict(X)
        f_Z = f.predict(Z)
        
        relevance_error = np.mean(f_X != f_Z)
        
        if self.console.verbose:
            self.console.log(f"Relevance Error (K={K})", f"{relevance_error:.4f}")
            
        return relevance_error
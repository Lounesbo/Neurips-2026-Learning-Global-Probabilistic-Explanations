import numpy as np
import pandas as pd
from classifier import Classifier
from km import KM
from spectral_explainer import SpectralExplainer
from timeit import default_timer as timer

class SyntheticMajorityClassifier:
    """A mock classifier that perfectly implements Majority(x0, x1, x2)."""
    def __init__(self, dimension=40):
        self.dimension = dimension

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X shape: (m, 40)
        # We only care about the first 3 columns
        sum_bits = X[:, 0] + X[:, 1] + X[:, 2]
        # Return 1 if majority are 1, else -1
        return np.where(sum_bits >= 2, 1, -1)

def main():
    d = 40
    print(f"--- Starting Synthetic Test (d={d}) ---")
    
    # 1. Initialize the procedural oracle
    # Instead of training a DT, we use the logic directly
    oracle = SyntheticMajorityClassifier(dimension=d)

    # 2. Run Parallel KM
    # We use a split_depth of 6 to saturate 61 workers (2^6 = 64 tasks)
    km_algo = KM(classifier=oracle, 
                 is_degree_bounded=True,
                 max_degree=3,  # We know the target is degree 3
                 is_sample_bounded=True,
                 max_training_samples=5000, 
                 threshold=0.2,
                 split_depth=6, 
                 variable_ordering_heuristic="influence",
                 verbose=True)
    
    km_algo.learn()
    print(km_algo)
    
    # 3. Estimate Fidelity Error
    m = 2000
    X_test = np.random.randint(2, size=(m, d))
    f_err = km_algo.fidelity_error(X_test)
    print(f"Estimated Fidelity Error: {f_err:.4f}")
    
    # 4. Run Spectral Explainer
    # It should identify features {0, 1, 2} as the explanation
    explainer = SpectralExplainer(classifier=oracle,
                                  hypothesis=km_algo.hypothesis, 
                                  is_bounded=True,
                                  max_size=3,
                                  is_pruned=True,
                                  threshold=0.05,
                                  verbose=True)
    
    explainer.explain()
    print(explainer)
    
    # 5. Estimate Relevance Error
    Z_test = np.random.randint(2, size=(m, d))
    r_err = explainer.relevance_error(X_test, Z_test)
    print(f"Estimated Relevance Error: {r_err:.4f}")

if __name__ == "__main__":
    main()
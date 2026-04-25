import numpy as np
import pandas as pd
from classifier import Classifier
from km import KM
from spectral_explainer import SpectralExplainer

def main():
    # 1. Create a synthetic 3-bit dataset
    # All possible inputs for d=3
    X_raw = np.array([[i >> j & 1 for j in range(3)] for i in range(8)])
    # Target: XOR of the first two bits (x0 ^ x1) mapped to {-1, 1}
    # x0=0, x1=0 -> 0 -> -1
    # x0=0, x1=1 -> 1 ->  1
    # x0=1, x1=0 -> 1 ->  1
    # x0=1, x1=1 -> 0 -> -1
    Y_raw = 2 * (X_raw[:, 0] ^ X_raw[:, 1]) - 1

    df_X = pd.DataFrame(X_raw, columns=['feat0', 'feat1', 'feat2'])
    df_Y = pd.DataFrame(Y_raw, columns=['Label'])

    # 2. Train a perfect classifier (Decision Tree)
    clf = Classifier(name="Decision Tree")
    clf.train(instances=df_X, labels=df_Y)
    
    # 3. Initialize KM
    km = KM(classifier=clf, 
            is_degree_bounded = False, 
            is_sample_bounded = True, 
            max_training_samples = 1000, 
            threshold=0.5, 
            variable_ordering_heuristic="lexicographic", 
            verbose = True)
    km.learn()
    print(km)

    # Sample size
    m = 1000

    X = np.random.randint(2, size=(m, 3)) 
    fidelity_error = km.fidelity_error(X)
    print(f"Estimated Fidelity Error: {fidelity_error:.4f}")
    
    # 4. Explanations
    explainer = SpectralExplainer(classifier=clf,
                                  hypothesis=km.hypothesis, 
                                  is_bounded = False,
                                  is_pruned = False,
                                  verbose=True)
    explainer.explain()
    print(explainer)
    Z = np.random.randint(2, size=(m, 3))
    relevance_error = explainer.relevance_error(X, Z)
    print(f"Estimated Relevance Error: {relevance_error:.4f}")

if __name__ == "__main__":
    main()
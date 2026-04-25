from matplotlib.pyplot import clf
import numpy as np
import pandas as pd
from classifier import Classifier
from km import KM
from explainer_km import KM_Explainer
import km
from sampler import Sampler
import sampler
from spectral_explainer import SpectralExplainer

def main():
     # 1. Create a synthetic 10-bit dataset
     # All possible inputs for d=10
     X_raw = np.array([[i >> j & 1 for j in range(10)] for i in range(1024)])
     # Target: Majority of the first 3 bits (x0, x1, x2) mapped to {-1, 1}
     sum_bits = X_raw[:, 0] + X_raw[:, 1] + X_raw[:, 2]
     Y_raw = np.where(sum_bits >= 2, 1, -1)

     df_X = pd.DataFrame(X_raw, columns=[f'feat{i}' for i in range(10)])
     df_Y = pd.DataFrame(Y_raw, columns=['Label'])

     # 2. Train a perfect classifier (Decision Tree)
     clf = Classifier(name="Decision Tree")
     clf.train(instances=df_X, labels=df_Y)

     # 3. Run KM
     km = KM(classifier=clf, 
             is_degree_bounded = False,
             is_sample_bounded = True,
             max_training_samples = 2000,
             threshold=0.3,
             variable_ordering_heuristic="lexicographic",
             verbose = True)
     km.learn()
     print(km)
     
     # 4. Estimate Fidelity Error
     m = 1000
     
     X = np.random.randint(2, size=(m, 10))
     fidelity_error = km.fidelity_error(X)
     print(f"Estimated Fidelity Error: {fidelity_error:.4f}")
     
     explainer = SpectralExplainer(classifier=clf,
                                  hypothesis=km.hypothesis, 
                                  is_bounded = False,
                                  is_pruned = False,
                                  verbose=True)
     explainer.explain()
     print(explainer)
     
     Z = np.random.randint(2, size=(m, 10))
     relevance_error = explainer.relevance_error(X, Z)
     print(f"Estimated Relevance Error: {relevance_error:.4f}")      
     


if __name__ == "__main__":
    main()

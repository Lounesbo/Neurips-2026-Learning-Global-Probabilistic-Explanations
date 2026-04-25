import numpy as np
from dataset import Dataset
from classifier import Classifier
from km import KM
from spectral_explainer import SpectralExplainer
from utils import Console

def main():
    console = Console(verbose = True)

    # Load and Binarize Tic-Tac-Toe Dataset
    data = Dataset(data_id=42192, nb_bins=4, verbose=True)
    data.setup()
    print(data) 

    # Train the Benchmark (using neural network)
    clf = Classifier(name="Random Forest")
    clf.train(instances=data.X, labels=data.Y)
    print(clf)

    # Run the KM Explainer
    km = KM(classifier=clf, 
            is_degree_bounded=True,
            max_degree=6,  
            is_sample_bounded=True,
            max_training_samples=5000, 
            threshold=0.1,
            split_depth=6, 
            variable_ordering_heuristic="influence",
            verbose=True)
    km.learn()
    print(km)

    # Sample size
    m = 10000

    X = np.random.randint(2, size=(m, data.nbBinFeatures))
    fidelity_error = km.fidelity_error(X)
    console.log("Estimated Fidelity Error", f"{fidelity_error:.4f}")
    
    explainer = SpectralExplainer(classifier=clf,
                                  hypothesis=km.hypothesis, 
                                  is_bounded = True,
                                  is_pruned = False,
                                  max_size = 7,
                                  verbose=True)
    explainer.explain()
    print(explainer)
    
    Z = np.random.randint(2, size=(m, data.nbBinFeatures))
    relevance_error = explainer.relevance_error(X, Z)
    console.log("Estimated Relevance Error", f"{relevance_error:.4f}")
    
    
if __name__ == "__main__":
    main()

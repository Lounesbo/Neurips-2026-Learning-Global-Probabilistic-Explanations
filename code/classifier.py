import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from utils import Console, Metrics
from typing import Tuple, Union, Dict

class Classifier:
    # Factory for models to ensure each instance gets a fresh model object
    MODELS = {
        "AdaBoost": AdaBoostClassifier,
        "Decision Tree": DecisionTreeClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "Linear SVM": LinearSVC,
        "Neural Network": MLPClassifier,
        "Random Forest": RandomForestClassifier,
    }

    def __init__(self, *, name: str, verbose: bool = False):
        self.console = Console(verbose=verbose)
        self.name = name
        
        if name not in self.MODELS:
            raise ValueError(f"Classifier '{name}' not found. Choose from: {list(self.MODELS.keys())}")
        
        # Instantiate the model with a fixed random state for reproducibility
        self.classifier = self.MODELS[name](random_state=0)
        
        # Metadata initialized to None/0 until training
        self.dimension = 0
        self.error = None

    def __str__(self):
        output = self.console.string("Classifier information", endl=True)
        output += self.console.string("Name", self.name, endl=True)
        output += self.console.string("Input Dimension", self.dimension, endl=True)
        
        # Handle cases where the model hasn't been trained yet
        error_val = f"{self.error:.4f}" if self.error is not None else "Not trained"
        output += self.console.string("Accuracy error", error_val, endl=True)
        return output

    def train(self, *, instances: pd.DataFrame, labels: pd.DataFrame) -> Tuple[int, float]:
        self.console.log(f"Training {self.name}...")
        
        X = instances.to_numpy()
        Y = labels.values.ravel()
        
        # Use context manager to catch warnings only during training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.classifier.fit(X, Y)
        
        # Compute stats
        predictions = self.classifier.predict(X)
        self.dimension = X.shape[1]
        self.error = Metrics.avg_zero_one_loss(labels=Y, predicted=predictions)
        
        return self.dimension, self.error

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[float, np.ndarray]:
        """
        Predict the value of a single instance or multiple instances.
        """
        # Convert DataFrame to numpy if necessary
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        # Scikit-learn models expect 2D arrays (samples, features)
        # If X is a single 1D vector, we reshape it to (1, n_features)
        if X.ndim == 1:
            prediction = self.classifier.predict(X.reshape(1, -1))
            return prediction[0]
        
        return self.classifier.predict(X)

import numpy as np
import pandas as pd
import dataset 
import classifier
import sampler
# import regressor
# import analyzer
from utils import Console

console = Console(verbose = True)
dataId = 37
# dataId = 44032
nbBins = 3
# regressorName = "MLP Regressor"
# spread = 1.0
# samplerName = "Distance"

console.log("Setup dataset")
data = dataset.Dataset(data_id = dataId, nb_bins = nbBins, verbose = True)
dataName = data.setup()
console.log("Dataset statistics")
print(data)

console.log("Setup classifier")
classifierName = "Neural Network"
clf = classifier.Classifier(name = classifierName, verbose = True)
clf.train(instances = data.X, labels = data.Y)
console.log("Classifier statistics")
print(clf)

# console.log("Setup sampler")
# samplerName = "Uniform"
# sampler = sampler.Sampler(features = data.X.columns, distribution = samplerName, verbose = True)
# sampler.setup()
# # Generate 10 samples
# nb_samples = 10
# samples = sampler.sample(nb_samples)
# print(samples)
# console.log("Sampler statistics")
# print(sampler)
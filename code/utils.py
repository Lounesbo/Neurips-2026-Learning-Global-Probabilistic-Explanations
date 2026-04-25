import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class Console:
    def __init__(self, *, decimals = 3, length = 40, verbose = True):
        self.decimals = decimals
        self.length = length
        self.verbose = verbose

    def log(self, first, second = None, endl = False):
        if not self.verbose:
            return
        if isinstance(first,str):
            print(self.string(first, second, endl))
        else:
            if isinstance(first,dict):
                print(self.dictionary(first))
        
    def dictionary(self, obj):
        str = ''
        for first, second in obj.items():
            str += self.string(f"{first}", f"{second}", endl = True)
        return str 
    
    def string(self, first, second = None, endl = False):
        if second is None:
            str = first.ljust(self.length,'.')
        else:            
            str = first.ljust(self.length - 1,'.') + ': '
            if isinstance(second, float):
                str += f"{second:.{self.decimals}f}"
            else:
                str += f"{second}"
        if endl:
            str += "\n"
        return str

class Convert:
    # Convert sparse vector to dictionary
    @staticmethod
    def vec_to_dict(vector:np.ndarray, columns:pd.Index) -> dict:
        indices, = np.nonzero(vector)
        dictionary = {}
        for j in indices:
            dictionary[columns[j]] = vector[j]
        return dictionary

    # Convert list of tuples (feature, val) to vector of vals
    @staticmethod
    def list_to_vec(list: list, columns:pd.Index) -> np.ndarray:
        vec = np.zeros(len(columns), dtype='float64')
        for pair in list:
            i = columns.get_loc(pair[0])
            vec[i] = pair[1]
        return vec
    
class Metrics:
    # Averaged zero-one loss error  
    @staticmethod
    def avg_zero_one_loss(*, labels:np.ndarray, predicted:np.ndarray) -> float:
        return 1 - accuracy_score(labels, predicted)  
    
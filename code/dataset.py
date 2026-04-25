import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from utils import Console

class Dataset:
    OPENML_DATASETS = {
        24: "Mushroom", 
        37: "Pima Indians Diabetes", 
        50: "Tic-Tac-Toe",
        56: "Vote", 
        335: "Monks", 
        336: "SPECT Heart",
        40670: "DNA Splice",
        40683: "Postoperative",
        42192: "Compass", 
        43595: "Loan-Predication", 
        43891: "Communities and Crime",
        43944: "Heart Disease", 
        43955: "Poker", 
        45064: "Insurance",
        45578: "California Housing",
    }

    def __init__(self, data_id=-1, data_name="", data_version=1, nb_bins=4, verbose=True):
        self.console = Console(verbose=verbose)
        self.dataId = data_id
        self.dataName = data_name or self.OPENML_DATASETS.get(data_id, f"Id{data_id}")
        self.dataVersion = data_version
        self.nbBins = nb_bins

        # Metadata
        self.nbMissingValues = 0
        self.nbRawFeatures = 0
        self.nbCatFeatures = 0
        self.nbNumFeatures = 0
        self.nbBinFeatures = 0
        self.nbLabels = 0
        self.majorityClass = None
        
        self.groups = []
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.rawData = pd.DataFrame()
        self.rawLabels = pd.Series(dtype=object)

    def __str__(self):
        stats = {
            "ID": self.dataId,
            "Name": self.dataName,
            "# Missing Values": self.nbMissingValues,
            "# Raw Features": self.nbRawFeatures,
            "# Categorical Features": self.nbCatFeatures,
            "# Numerical Features": self.nbNumFeatures,
            "# Binary Features": self.nbBinFeatures,
            "# Instances": self.X.shape[0],
            "# Labels": self.nbLabels,
            "Target (+1)": self.majorityClass
        }
        res = self.console.string("Dataset Information", endl=True)
        for k, v in stats.items():
            res += self.console.string(k, v, endl=True)
        return res
    
    def setup(self) -> str:
        self.getData()
        self.cleanData()
        self.binarizeData()
        self.encodeLabels()
        return self.dataName

    def getData(self) -> None:
        self.console.log(f"Loading dataset: {self.dataName}")
        fetch_kwargs = {"data_id": self.dataId} if self.dataId != -1 else {"name": self.dataName, "version": self.dataVersion}
        X, y = fetch_openml(**fetch_kwargs, parser='auto', as_frame=True, return_X_y=True)
        self.rawData = X.toarray() if hasattr(X, "toarray") else X
        self.rawLabels = y
        self.nbRawFeatures = self.rawData.shape[1]

    def cleanData(self) -> None:
        self.console.log("Cleaning and imputing missing values")
        missing_strings = ['?', '', '.', 'nan', 'na', 'none', 'null']
        self.rawData = self.rawData.replace(missing_strings, np.nan)
        self.nbMissingValues = self.rawData.isnull().values.sum()

        if self.nbMissingValues > 0:
            imp = SimpleImputer(strategy="most_frequent")
            self.rawData = pd.DataFrame(
                imp.fit_transform(self.rawData),
                columns=self.rawData.columns,
                index=self.rawData.index
            )

    def binarizeData(self) -> None:
        self.console.log("Binarizing features...")
        temp_df = self.rawData.loc[:, self.rawData.nunique() > 1].copy()
        feature_parts = []
        current_idx = 0
        
        cat_cols = temp_df.select_dtypes(exclude=['number']).columns.tolist()
        low_card_num_cols = [c for c in temp_df.select_dtypes(include=['number']).columns 
                             if temp_df[c].nunique() <= self.nbBins]
        
        self.nbCatFeatures = len(cat_cols)
        for col in (cat_cols + low_card_num_cols):
            df_one_hot = pd.get_dummies(temp_df[col], prefix=col, dtype=int)
            feature_parts.append(df_one_hot)
            num_new_cols = df_one_hot.shape[1]
            self.groups.append(list(range(current_idx, current_idx + num_new_cols)))
            current_idx += num_new_cols

        cont_cols = [c for c in temp_df.columns if c not in (cat_cols + low_card_num_cols)]
        self.nbNumFeatures = len(cont_cols)

        if cont_cols:
            discretizer = KBinsDiscretizer(n_bins=self.nbBins, encode='onehot-dense', strategy='uniform')
            bin_data = discretizer.fit_transform(temp_df[cont_cols]).astype(int)
            df_bin_num = pd.DataFrame(
                bin_data, 
                columns=discretizer.get_feature_names_out(cont_cols),
                index=temp_df.index
            )
            feature_parts.append(df_bin_num)
            for bins_count in discretizer.n_bins_:
                self.groups.append(list(range(current_idx, current_idx + bins_count)))
                current_idx += bins_count

        self.X = pd.concat(feature_parts, axis=1)
        self.nbBinFeatures = self.X.shape[1]

    def encodeLabels(self) -> None:
        """
        Reduces the problem to binary classification: 
        Majority Class -> +1, All others -> -1.
        """
        self.console.log("Encoding labels to {-1, 1} (Majority-vs-Rest)...")
        
        # 1. Determine the majority class
        counts = self.rawLabels.value_counts()
        self.majorityClass = counts.idxmax()
        count_val = counts.max()
        total = len(self.rawLabels)
        
        self.console.log(f"Majority Class: '{self.majorityClass}' ({count_val}/{total} instances)")

        # 2. Perform the mapping: Majority -> 1, Rest -> -1
        binary_labels = np.where(self.rawLabels == self.majorityClass, 1, -1)
        
        self.Y = pd.DataFrame(binary_labels, columns=["Label"], index=self.rawData.index)
        
        # Metadata update: Now it's always a binary problem for the algorithm
        self.nbLabels = 2

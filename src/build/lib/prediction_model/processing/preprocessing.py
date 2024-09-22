from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config


class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.mean_dict = {}
        for col in self.columns:
            self.mean_dict[col] = X[col].mean()
        return self    

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.mean_dict[col])
        return X    


class modeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.mode_dict = {}
        for col in self.columns:
            self.mode_dict[col] = X[col].mode()[0]
        return self    

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.mode_dict[col])
        return X    
    

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self    

    def transform(self, X):
        X = X.copy()
        X.drop(columns=self.columns_to_drop, inplace=True)
        return X        
    

class DomainProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_modify=None, column_to_add=None):
        self.column_to_modify = column_to_modify
        self.column_to_add = column_to_add

    def fit(self, X, y=None):
        return self    

    def transform(self, X):
        X = X.copy()
        for col in self.column_to_modify:
            X[col] = X[col] + X[self.column_to_add]
        return X        

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.label_encoder_dict = {}
        for col in self.columns:
            t = X[col].value_counts().sort_values(ascending=True).index
            self.label_encoder_dict[col] = {k:i for i,k in enumerate(t)}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.label_encoder_dict[col])
        return X    
            

class LogTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self    

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = np.log(X[col])
        return X    
                
import numpy as np
from pathlib import Path
import os
import sys


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))


from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression



classification_pipeline = Pipeline(
    [
        ('MeanImputation', pp.MeanImputer(columns=config.NUM_FEATURES)),
        ('ModeImputation', pp.modeImputer(columns=config.CAT_FEATURES)),
        ('DomainProcessing', pp.DomainProcessing(column_to_modify=config.FEATURE_TO_MODIFY, column_to_add=config.FEATURE_TO_ADD)),
        ('DropFeatures', pp.DropColumns(columns_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder', pp.CustomLabelEncoder(columns=config.FEATURES_TO_ENCODE)),
        ('LogTransform', pp.LogTransforms(columns=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier', LogisticRegression(random_state=0))
    ]
)
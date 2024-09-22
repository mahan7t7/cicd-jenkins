import pandas as pd 
import joblib

from pathlib import Path
import os
import sys


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))


from prediction_model.config import config


def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f'The model has been saved under the name {config.MODEL_NAME}')

def load_pipeline(pipeline_to_load):
    load_path = os.path.join(config.SAVE_MODEL_PATH, pipeline_to_load)
    model_loaded = joblib.load(load_path)
    print('The model has been loaded') 
    return model_loaded    
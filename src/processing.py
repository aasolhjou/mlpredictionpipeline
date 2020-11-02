#import packages
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

import config

#load dataset
def load_dataset(*, file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data

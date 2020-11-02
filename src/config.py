import pathlib
from pipeline import model_pipelines

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = 'main_df.csv'
TRAINING_DATA_FILE = 'main_df.csv'

TARGET = 'price'

FEATURES = ['location', 'year', 'km', 'fuel', 'gears',
            'owners', 'kmpl', 'engine_size', 'power',
            'seats']

TOP5_FEATURES = ['power', 'engine_size', 'gears', 'fuel', 'kmpl']

BEST_MODEL = model_pipelines['ExtraTrees']

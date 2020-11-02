#import packages
import config
from pipeline import model_pipelines
from processing import load_dataset
import _pickle as pickle
from yellowbrick.regressor import PredictionError, ResidualsPlot, AlphaSelection

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import  LinearRegression, LassoCV, RidgeCV, ElasticNetCV, SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor

#load data
visualizations = load_dataset(file_name=config.TRAINING_DATA_FILE)

#set X and y
#adjust X based on feature set to use from config.py (TOP5_FEATURES or FEATURES)
X = visualizations[config.TOP5_FEATURES]
y = visualizations[config.TARGET]

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#yellowbrick ResidualsPlotVisualization visual
visualizer = ResidualsPlot(config.BEST_MODEL)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show(outpath="visualizations/ResidualsPlotVisualization.pdf")
visualizer.show(outpath="visualizations/ResidualsPlotVisualization.png")
visualizer.show()

#yellowbrick prediction error visual
visualizer = PredictionError(config.BEST_MODEL)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show(outpath="visualizations/PredictionErrorVisualization.pdf")
visualizer.show(outpath="visualizations/PredictionErrorVisualization.png")
visualizer.show()

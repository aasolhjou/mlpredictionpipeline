#import packages
import _pickle as pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import  LinearRegression, LassoCV, RidgeCV, ElasticNetCV, SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from yellowbrick.regressor import PredictionError, ResidualsPlot, AlphaSelection
from sklearn.metrics import mean_squared_error as mse, r2_score, accuracy_score


import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline

#Creating A Pipeline for each model

#Dummy Regressor
pipe_dumreg = Pipeline([('scale', StandardScaler()),('model',
DummyRegressor())])

#Linear LinearRegression
pipe_linreg = Pipeline([('scale', StandardScaler()),('model',
LinearRegression())])

#Lasso CV Regression
pipe_lassoreg = Pipeline([('scale', StandardScaler()), ('model',
LassoCV())])

#Ridge CV Regression
pipe_ridgereg = Pipeline([('scale', StandardScaler()), ('model',
RidgeCV())])

#Elastic Net CV Regression
pipe_elasticnetreg = Pipeline([('scale', StandardScaler()), ('model',
ElasticNetCV())])

#Linear SVR Regression
pipe_linsvrreg = Pipeline([('scale', StandardScaler()), ('model',
LinearSVR())])

#NU SVR Regression
pipe_nusvrreg = Pipeline([('scale', StandardScaler()), ('model',
NuSVR())])

#Random Forest Regression
pipe_rfreg = Pipeline([('scale', StandardScaler()), ('model',
RandomForestRegressor())])

#AdaBoost Regression
pipe_adaboostreg = Pipeline([('scale', StandardScaler()), ('model',
AdaBoostRegressor())])

#Bagging Regression
pipe_bagreg = Pipeline([('scale', StandardScaler()), ('model',
BaggingRegressor())])

#Extra Trees Regression
pipe_xtreg = Pipeline([('scale', StandardScaler()), ('model',
ExtraTreesRegressor())])

#Gradient Boosting Regression
pipe_gbreg = Pipeline([('scale', StandardScaler()), ('model',
GradientBoostingRegressor())])

#Create list of model names and pipelines

models = ['DummyRegressor', 'LinearRegression', 'LassoCV', 'RidgeCV',
        'ElasticNetCV','LinearSVR', 'NuSVR', 'RandomForest','AdaBoost',
        'BaggingRegression', 'ExtraTrees','GradientBoosting']

pipelines = [pipe_dumreg, pipe_linreg, pipe_lassoreg, pipe_ridgereg,
            pipe_elasticnetreg, pipe_linsvrreg, pipe_nusvrreg,
            pipe_rfreg, pipe_adaboostreg, pipe_bagreg,
            pipe_xtreg, pipe_gbreg]


model_pipelines = dict(zip(models, pipelines))

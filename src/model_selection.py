#import packages
import pathlib
import numpy as np
import pandas as pd
import _pickle as pickle

from pathlib import Path


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import  LinearRegression, LassoCV, RidgeCV, ElasticNetCV, SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from yellowbrick.regressor import PredictionError, ResidualsPlot, AlphaSelection
from sklearn.metrics import mean_squared_error as mse, r2_score, accuracy_score


from processing import load_dataset
import config
from pipeline import model_pipelines

#set save path for saved trained_model.pkl file
root = Path(".")

path = root / "trained_models" / "trained_model.pkl"

#load in dataset
final_df = load_dataset(file_name=config.TRAINING_DATA_FILE)

#set X and y
#adjust X based on feature set to use from config.py(TOP5_FEATURES or FEATURES)
X = final_df[config.TOP5_FEATURES]
y = final_df[config.TARGET]

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#loop through model pipeline and fit each pipeline to training data
for name, pipe in model_pipelines.items():
    print(pipe)
    pipe.fit(X_train, y_train)

print("Models Now Trained")

print("Calculating Cross Val Score to check for Overfitting")

print("Cross Val Scores")

#create dictionary of cross val scores for each pipeline
models_cross_val = {}

seed = 7
kfold = KFold(n_splits=10, random_state=seed)

#loop through each pipeline and calculate cross val score
for name, pipe in model_pipelines.items():

    cross_val = cross_val_score(pipe, X_train, y_train)

    cross_val = cross_val.mean()

    models_cross_val[name] = cross_val

    print('\n' + "{}: {}".format(name, cross_val))

print("Best Model by Cross Val Score: ")

#select best cross val score
for i in sorted(models_cross_val, key=models_cross_val.get, reverse=True)[:1]:
    print (i ,models_cross_val[i])

print("Calculating R2 and RMSE Scores....")

print("R2 Scores: ")

#create dictionary of r2 scores for each pipeline
models_r2 = {}

#loop through each pipeline and calculate r2 score
for name, pipe in model_pipelines.items():

    r2 = r2_score(y_test, pipe.predict(X_test))

    models_r2[name] = r2

    print('\n' + "{}: {}".format(name, r2))

print("Best Model by R2: ")

#select best r2 score
for i in sorted(models_r2, key=models_r2.get, reverse=True)[:1]:
    print (i ,models_r2[i])

print("RMSE Scores: ")

#create dictionary of rmse scores for each pipeline
models_rmse = {}

#loop through each pipeline and calculate rmse score
for name, pipe in model_pipelines.items():

    rmse_score = np.sqrt(mse(y_test, pipe.predict(X_test)))

    models_rmse[name] = rmse_score

    print('\n' + "{}: {}".format(name, rmse_score))

print("Best Model by RMSE: ")

#select best r2 score
for i in sorted(models_rmse, key=models_rmse.get, reverse=False)[:1]:
    print (i ,models_rmse[i])

#first loop through r2 scores
for i in sorted(models_r2, key=models_r2.get, reverse=True)[:1]:
#then loop through RMSE scores
    for j in sorted(models_rmse, key=models_rmse.get, reverse=False)[:1]:
#select model that scored highest in both
        if i == j:
            print("Best Model is: " + j)
#show model input into config.py file for model_visualizer
            if j in model_pipelines.keys():
                print("Input '{}' into BEST_MODEL in config.py and run model_visualizer.py".format(j))
#save trained model to .pkl file
                trained_model = open(path, 'wb')
                trained_model = pickle.dump(model_pipelines[j], trained_model)

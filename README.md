# ML Prediction Pipeline - Used car sale prices in India. 

The following repository contains scripts to clean data and select the most
important features from data of used car sales in India.
The cleaned data is split into training and testing sets,
and passed into a ML pipeline. It is then funneled through 11 regression models,
and a dummy regression model, where each model is checked for over-fitting.
Finally the model with the best R2 and RMSE scores is saved as a .pkl file,
and is passed into a model-visualizer.

## Installation

Please complete the following steps once in the correct directory:

1. Install all required packages(once in regression_model folder):
```bash
pip install -r requirements.txt
```

2. Navigate to src folder and run data_cleaning.py.

```bash
python data_cleaning.py
```
This will clean the data and output  main_df.csv.

3. Navigate to src folder and run feature_selection.py.

```bash
python feature_selection.py
```
This will select the 5 best features from the data,
and save visualizations of price vs. each feature
to the visualizations folder.

4. Navigate to src folder and run model_selection.py.

```bash
python model_selection.py
```

This will funnel the data through a ML pipeline and
select the best model based on their R2 and RMSE scores. If you would like to run
on all of the model features, set X inside model_selection.py as such:

```python
X = final_df[config.FEATURES]
```

5. Navigate to src folder and run model_visualizer.py.

This will create a Residuals Plot Visualization, and a Prediction Error Visualization,
utilizing the yellowbrick library.

```bash
python model_visualizer.py
```
The visualizations will pop up on the screen, and be saved to the visualizations
folder as PDFs.

If you would like to run the model_visualizer
on all of the model features, set X inside model_visualizer.py as such:

```python
X = final_df[config.FEATURES]
```

## Finding - Feature Selection
There were 11 features in total. The new price feature was dropped as it had
too many missing values. The name and wear_factor features were also dropped, as
each value for those features was unique to each row, and thus not needed.

In order to find the 5 best features correlated
with price, a correlation matrix was created, and the 5 most correlated features
were selected from there.

The top 5 most correlated features were power, engine_size, gears, fuel, kmpl.

This makes sense, as a more powerful engine usually means that it is a nicer car,
and thus be a car that is more in demand. Gears also makes sense, as a buyer will
have a strong preference to driving automatic or manual. Similarly, a buyer will
either want a car that runs on petrol, or diesel fuel. The kmpl is also important,
as a buyers usually take into consideration a car's fuel consumption.

## Findings - Predictive Model Selection

The ML pipeline is constructed such that the data is
scaled before the models are run. Each model used their default parameters. The
following models were used in the pipeline:

### Linear Models
1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. Elastic Net Regression

### SVM Models
1. Linear SVR
2. Nu SVR

### Ensemble Models
1. Random Forest Regression
2. Ada Boost Regression
3. Bagging Regression
4. Extra Trees Regression
5. Gradient Boosting

Overall, the Ensemble Models performed the best.

## Findings - Model Performance

Each model was first tested for over-fitting by using cross_val_score.

Each model was then tested for an R2 and RMSE score.

When tested with the top 5 features only, the best model was split between
Random Forest, Bagging Regression, and Extra Trees. Both Extra Tree and Random
models are similar to one another, but Random Forest uses a greedy algorithm to
select an optimal split point, while Extra Trees algorithm selects a
split point at random. In Random Forest, a subset of the features are selected
at random, where Bagging Regression all features are considered.

When tested on all features, the best model varied between Random Forest,
Bagging Regression, Extra Trees, and Gradient Boosting.

The models all performed worse when used on the top 5 features, compared to
utilizing all of the features. This is most likely due to the fact that the
models had less data to work with. Often there is a trade off between model
performance and run time. A diagnostic test on run time could be run to assess
the benefit of using less features.

To improve Model Performance, a grid search could be used on the
best model(s). This would test for the optimal parameters for each model in order
to find the best performing ones. However, this is computationally heavy,
so for that reason it was not included.

## Model Weaknesses

If a competitor to the used car sales company knew this algorithm, there are
several ways they could leverage this information to take advantage of
their business model. As the algorithm weighs power and engine size
as the two most important factors, a competitor could attempt to sell an older
car with a beefed up engine. Since the algorithm does not take
name or wear factor into account, this car could be an old piece of junk
that happened to have a strong engine in it. This would cause the used car sales
company to predict a much higher sales price than what would realistically sell.

#import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read in data set and check data set
cars = pd.read_csv('datasets/Used car sale price estimation.csv', index_col = 0)
cars.shape
cars.head()
cars.info()

#drop unused index
cars.drop('index', axis=1, inplace=True)
#rename columns
cars.columns = ['name', 'location', 'year', 'km', 'fuel', 'gears', 'owners', 'kmpl', 'engine_size', 'power', 'seats', 'new_price', 'price', 'wear_factor']
cars.dtypes
cars.isna().any()
cars.isnull().sum()

#drop new_price column
cars.drop('new_price', axis=1, inplace=True)
cars = cars.dropna()

#check name column and drop name column
cars.name.value_counts().sort_index()
cars.drop('name', axis=1, inplace=True)

#check location column and map to integers to change datatype
cars.location.value_counts().sort_index()
cars.shape
cars['location'] = cars.location.map({'Ahmedabad': 1, 'Bangalore': 2, 'Chennai': 3, 'Coimbatore': 4, 'Delhi': 5, 'Hyderabad': 6, 'Jaipur': 7, 'Kochi': 8, 'Kolkata': 9, 'Mumbai': 10, 'Pune': 11})
cars['location'] = cars['location'].astype(int)

#check year column, drop 1804 years.
cars.year.value_counts().sort_index()
cars[cars['year'] == 1804]
cars.drop([583, 1142, 1246, 1579, 1688, 1809, 2344, 2592, 4277, 4361, 4962], axis=0, inplace=True)
cars.shape

#check km column, filter out large outliers
cars.km.value_counts().sort_index()
cars[cars['km'] > 250000]
cars[cars['km'] < 1500]
cars = cars[(cars['km'] > 1500) & (cars['km'] < 250000)]
cars.shape

#check fuel column, filter to only diesel and petrol fuel, change datatype
cars.fuel.value_counts()
cars = cars[(cars['fuel'] == 'Diesel') | (cars['fuel'] == 'Petrol')]
cars['fuel'] = cars.fuel.map({'Diesel':1, 'Petrol':0})
cars['fuel'] = cars['fuel'].astype(int)
cars.shape

#check gears column, map to integers, change datatype
cars.gears.value_counts().sort_index()
cars['gears'] = cars.gears.map({'Automatic': 1, 'Manual': 0})
cars['gears'] = cars['gears'].astype(int)

#check owners columns, map to integers, change datatype
cars.owners.value_counts().sort_index()
cars['owners'] = cars.owners.map({'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4})
cars['owners'] = cars['owners'].astype(int)

#check kmpl column, filter out null values, change datatype
cars.kmpl.value_counts().sort_index()
cars = cars[~cars.kmpl.str.contains('0.0 kmpl')]
cars['kmpl'] = cars['kmpl'].str.replace('kmpl','')
cars['kmpl'] = cars['kmpl'].astype(float)

#check engine_size column, filter out cc, change datatype
cars.engine_size.value_counts().sort_index(ascending=True)
cars['engine_size'] = cars['engine_size'].str.replace('CC','')
cars['engine_size'] = cars['engine_size'].astype(float)

#check power column, filter out null values, change datatype
cars['power'] = cars['power'].str.replace('bhp', '')
cars['power'].value_counts().sort_index()
cars = cars[~cars.power.str.contains('null')]
cars['power'] = cars['power'].astype(float)

#check seats column, filter out high seat numbers
cars.seats.value_counts().sort_index()
cars[cars['seats'] > 8]
cars.drop([917, 1347, 1907, 2312, 2359, 2575], axis=0, inplace=True)

#check price coulmn, multiple by 100000 to convert lakh to rupees, filter data
cars.price = cars.price * 100000
cars.price.value_counts().sort_index()
cars = cars[(cars['price'] < 9000000)]

#check wear_factor column, drop column
cars.wear_factor.value_counts().sort_index()
cars.drop('wear_factor', axis=1, inplace=True)
cars.shape

#check data again
cars.info()
cars.isnull().sum()
cars.describe()

clean_cars = cars
clean_cars

print("done cleaning")

#send to new csv file
clean_cars.to_csv('datasets/main_df.csv')

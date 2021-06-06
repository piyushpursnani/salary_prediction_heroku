import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('hiring.csv')
print(dataset)

dataset.experience.fillna(0, inplace = True)
dataset.test_score.fillna(dataset.test_score.mean(), inplace = True)

X = dataset.iloc[:,:3]


y = dataset.iloc[:,:1]

regressor = LinearRegression()

regressor.fit(X,y)

print('model training is done')

joblib.dump(regressor,'model.pkl')

print(regressor.predict([[1,7,9]]))

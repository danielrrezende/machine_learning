import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

path = 'melb_data.csv'
data = pd.read_csv(path)
data = data.dropna(axis=0)

y = data.Price
print("y.shape  %d" %(y.shape))

X = data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]
##X = X[:, None]
print("X.shape", X.shape)
      
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("train_X.shape", (X_train.shape))
print("train_y.shape", (y_train.shape))
print("val_X.shape", (X_test.shape))
print("val_y.shape", (y_test.shape))

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

val_pred_prices = model.predict(X_test)
print('val_pred_prices', val_pred_prices)

mean_absolute_error(y_test, val_pred_prices)
print('mean_absolute_error', mean_absolute_error)

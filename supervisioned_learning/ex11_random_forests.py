import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Path of the file to read
iowa_file_path = 'train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
##features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes = 71, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# # Define the model. Set random_state to 1
rf_model = RandomForestRegressor(max_leaf_nodes = 318, random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

##var = 300000
##best_leaf = 0
##
### Using best value for max_leaf_nodes
##def get_best_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
##    iowa_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=1)
##    iowa_model.fit(train_X, train_y)
##    val_predictions = iowa_model.predict(val_X)
##    val_mae = mean_absolute_error(val_predictions, val_y)
##    return(val_mae)
##
##for max_leaf_nodes in range(2,5000):
##    my_mae = get_best_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
##    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
##    if(my_mae < var):
##        var = my_mae
##        best_leaf = max_leaf_nodes
##print("The lowest Mean Absoule Error found is %d and the best leaf is %d" %(var,best_leaf))



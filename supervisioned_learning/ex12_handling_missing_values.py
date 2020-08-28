import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Path of the file to read
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
#home_data = data.dropna(axis=0)

# Create target object and call it y
y = home_data.SalePrice

# create all predictors exclude 'salesprice'
y_predictors = home_data.drop(['SalePrice'], axis=1)
# create only numeric predictors
X = y_predictors.select_dtypes(exclude=['object'])


# Split into validation and training data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)
# print("X_train.shape", (X_train.shape))
# print("y_train.shape", (y_train.shape))
# print("X_test.shape", (X_test.shape))
# print("y_test.shape", (y_test.shape))

# # Define the model. Set random_state to 1
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_test)
    val_mae = mean_absolute_error(y_test, val_predictions)
    #print("Validation MAE for Random Forest Model: {:,.0f}".format(val_mae))
    return val_mae

cols_with_missing = [col for col in X_train.columns
                                 if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
print("reduced_X_train.shape", (reduced_X_train.shape))
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("reduced_X_test.shape", (reduced_X_test.shape))

print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
print("imputed_X_train.shape", (imputed_X_train.shape))
imputed_X_test = my_imputer.transform(X_test)
print("imputed_X_test.shape", (imputed_X_test.shape))
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))





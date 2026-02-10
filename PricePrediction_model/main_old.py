import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. load the dataset
housing = pd.read_csv("housing.csv")

#2. Create a stratfied Train/test set

housing["income_cat"]= pd.cut(housing["median_income"] ,
                 bins = [0.0 ,1.5 ,3.0 , 4.5 ,6.0 , np.inf] ,
                 labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2 , random_state=42)

for train_index , test_index in split.split(housing , housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis = 1)
    strat_test_set = housing.loc[test_index].drop("income_cat" , axis = 1)

#working on copy of data
housing = strat_train_set.copy()

# 3. Seprate features and labels
housing_labels = housing["median_house_value"].copy()
housing =housing.drop("median_house_value" ,axis = 1)

print(housing , housing_labels)

#4. seperate numerical and categorial values
num_attribs = housing.drop("ocean_proximity",axis = 1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. pipeline

num_pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy="median")),
    ("scaler" , StandardScaler()),
])

cat_pipeline = Pipeline([
    ("onehot" , OneHotEncoder(handle_unknown = "ignore"))
])

# 6. constructing full Pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline , num_attribs),
    ("cat", cat_pipeline , cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# 7. Training the model


#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared , housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmses = -cross_val_score(lin_reg , housing_prepared , housing_labels , scoring= "neg_root_mean_squared_error" , cv = 10)
# print(f"root mean square error for Decision Tree Regressor is {dec_rmse}")
print(pd.Series(lin_rmses).describe())


# Decision Tree Regressor
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared , housing_labels)
dec_preds = lin_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_labels , dec_preds)
dec_rmses = -cross_val_score(dec_reg , housing_prepared , housing_labels , scoring= "neg_root_mean_squared_error" , cv = 10)
# print(f"root mean square error for Decision Tree Regressor is {dec_rmse}")
print(pd.Series(dec_rmses).describe())

# Random Forest Classifier
random_forest_reg = RandomForestClassifier()
random_forest_reg.fit(housing_prepared , housing_labels)
random_forest_preds = lin_reg.predict(housing_prepared)
random_forest_rmses = -cross_val_score(random_forest_reg , housing_prepared , housing_labels , scoring= "neg_root_mean_squared_error" , cv = 10)
# print(f"root mean square error for Decision Tree Regressor is {dec_rmse}")
print(pd.Series(random_forest_rmses).describe())
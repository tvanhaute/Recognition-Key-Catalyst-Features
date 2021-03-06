import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# read data
data = pd.read_csv("OCM-NguyenEtAl.csv")

# convert M1, M2 and M3 into numeric values
le = preprocessing.LabelEncoder()
data["M1"] = le.fit_transform(data["M1"])
data["M2"] = le.fit_transform(data["M2"])
data["M3"] = le.fit_transform(data["M3"])

# define labels, features and dropped features
predict = ["CH4_conv", "C2y", "C2H6y", "C2H4y", "COy", "CO2y"]
categorical_columns = ["M1", "M2", "M3", "Support_ID"]
numerical_columns = ["M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "M1_mol", "CT"]
features = categorical_columns + numerical_columns
dropped = ["CH4_conv", "C2y", "C2H6y", "C2H4y", "COy", "CO2y", "C2s", "C2H6s", "C2H4s", "COs", "CO2s", "Name",
           "M1_atom_number", "M2_atom_number", "M3_atom_number", "Support", "CH4/O2", "Total_flow", "M1_mol%",
           "M2_mol%", "M3_mol%"]

# calculate the M1 mol feature as it is missing in the original dataset
data["M2_mol%"] = data["M2_mol%"] + 1e-7 # to avoid divide by zero errors
data["M1_mol"] = data["M1_mol%"] * (data["M2_mol"] + data["M3_mol"]) / (data["M2_mol%"] + data["M3_mol%"])

# define dataframes to store feature importances in
columns = [feature for feature in features]
columns.append("Train score")
columns.append("Test score")

df = pd.DataFrame(columns=columns)
df1 = pd.DataFrame(columns=columns)

voorspel = predict[1]  # choose yield to predict

# make arrays of the label and features and standardize them
X = data[features]
X = preprocessing.StandardScaler().fit_transform(X)
y = np.array(data[voorspel])
y = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
y = np.ravel(y.reshape(-1, 1))

# loop that performs multiple dataset splits, constructs models based on these splits, selects the best model and stores the feature importances in the dataframes
number_of_splits = 10
random_states = [0, 42]
for l in range(number_of_splits):
    # split data into test and validation + training data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(x_train, y_train,
                                                                                        test_size=0.11)

    Validation_score = 0
    Train_score = 0
    Test_score = 0

    for random_state in random_states:
        forest = RandomForestRegressor(n_estimators=50, random_state=random_state)
        forest.fit(x_train, y_train)

        if (forest.score(x_validate, y_validate) > Validation_score):
            Validation_score = forest.score(x_validate, y_validate)
            Train_score = forest.score(x_train, y_train)
            Test_score = forest.score(x_test, y_test)

            importances = forest.feature_importances_

            result = permutation_importance(forest, x_test, y_test, n_repeats=10)
            importances_bis = result.importances_mean

    df_entry = []
    df_entry1 = []

    for i in range(X.shape[1]):
        df_entry.append(importances[i])
        df_entry1.append(importances_bis[i])

    df_entry.append(Train_score)
    df_entry.append(Test_score)

    df_entry1.append(Train_score)
    df_entry1.append(Test_score)

    df.loc[l] = df_entry
    df1.loc[l] = df_entry1

# print dataframes to csv files
df.to_csv('Results RFR impurity based.csv')
df1.to_csv('Results RFR permutation based.csv')

import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

# read data
data = pd.read_csv("OCM-NguyenEtAl.csv")

# convert M1, M2 and M3 into numeric values, superfluous and only included because of me working with multiple algorithms
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

# calculate M1 mol as it isn't included in the original dataset
data["M2_mol%"] = data["M2_mol%"] + 1e-7 # to avoid divide by zero errors
data["M1_mol"] = data["M1_mol%"] * (data["M2_mol"] + data["M3_mol"]) / (data["M2_mol%"] + data["M3_mol%"])

# define label
voorspel = predict[1]  # choose yield to predict

# define arrays for label and features, standardize label
X = data[features]
y = np.array(data[voorspel])
y = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
y = np.ravel(y.reshape(-1, 1))

# feature preprocessing
categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
numerical_pipe = Pipeline([
    ('imputer', preprocessing.StandardScaler())
])

preprocessing = ColumnTransformer(
    [('cat', categorical_encoder, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])

# define dataframe to store feature importances in
columns = [feature for feature in features]
columns.append("Train score")
columns.append("Test score")
df1 = pd.DataFrame(columns=columns)

# loop that performs multiple dataset splits and model constructions, chooses best model and saves feature importances for that model
number_of_splits = 10
C_values = [0.1, 1]
for l in range(number_of_splits):
    # split data into test and validation + training data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(x_train, y_train,
                                                                                        test_size=0.11)

    Validation_score = 0
    Train_score = 0
    Test_score = 0

    for C_value in C_values:
        svm = Pipeline([
            ('preprocess', preprocessing),
            ('regressor', SVR(C=C_value))
        ])

        svm.fit(x_train, y_train)

        if (svm.score(x_validate, y_validate) > Validation_score):
            Validation_score = svm.score(x_validate, y_validate)
            Train_score = svm.score(x_train, y_train)
            Test_score = svm.score(x_test, y_test)

            result = permutation_importance(svm, x_test, y_test, n_repeats=10)
            importances_bis = result.importances_mean

    df_entry1 = []

    for i in range(X.shape[1]):
        df_entry1.append(importances_bis[i])

    df_entry1.append(Train_score)
    df_entry1.append(Test_score)

    df1.loc[l] = df_entry1

# print feature importances to a csv file           
df1.to_csv("Results SVR yields Nguyen et al.csv")

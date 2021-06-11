import numpy as np
import pandas as pd
import sklearn
from tensorflow import keras
from sklearn import preprocessing
from sklearn.metrics import r2_score
from tensorflow.keras import layers
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# read data
data = pd.read_csv("../OCM-NguyenEtAl.csv")

# convert M1, M2 and M3 into numeric values, not necessary but artefact of me working with different algorithms
le = preprocessing.LabelEncoder()
data["M1"] = le.fit_transform(data["M1"])
data["M2"] = le.fit_transform(data["M2"])
data["M3"] = le.fit_transform(data["M3"])

# defining the possible labels, the features and the features that are dropped
predict = ["CH4_conv", "C2y", "C2H6y", "C2H4y", "COy", "CO2y"]
categorical_columns = ["M1", "M2", "M3", "Support_ID"]
numerical_columns = ["M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "M1_mol", "CT"]
features = categorical_columns + numerical_columns
dropped = ["CH4_conv", "C2y", "C2H6y", "C2H4y", "COy", "CO2y", "C2s", "C2H6s", "C2H4s", "COs", "CO2s", "Name",
           "M1_atom_number", "M2_atom_number", "M3_atom_number", "Support", "CH4/O2", "Total_flow", "M1_mol%",
           "M2_mol%", "M3_mol%"]

# the dataset doesn't contain the M1 mol feature so it has to be calculated first
data["M2_mol%"] = data["M2_mol%"] + 1e-7 # to avoid divide by zero errors
data["M1_mol"] = data["M1_mol%"] * (data["M2_mol"] + data["M3_mol"]) / (data["M2_mol%"] + data["M3_mol%"])

# choose C2 yield as label
voorspel = predict[1]  

# making arrays for the label and features, standardizing the label
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

# making a pandas dataframe to store feature importances
columns = [feature for feature in features]

columns.append("Train score")
columns.append("Test score")

df1 = pd.DataFrame(columns=columns)

# defining the early stop criterion for the neural net fitting
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# calculates regression R² score
def R2_scorer(ANN, x_test, y_test):
    y_pred = ANN.predict(x_test)
    score = r2_score(y_test,y_pred)
    return score

# builds a neural network with keras sequential
def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(53,)))
    model.add(layers.Dense(448, activation='relu'))
    model.add(layers.Dense(448, activation='relu'))
    model.add(layers.Dense(224, activation='relu'))
    model.add(layers.Dense(480, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['mae', 'mse'])
    return model

# makes models of multiple dataset splits, determines feature importances and saves them in the pandas dataframe
number_of_splits = 10
for l in range(number_of_splits):
    # split data into test and validation + training data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    model = build_model()

    ANN = Pipeline([
        ('preprocess', preprocessing),
        ('regressor', model)
    ])

    ANN.fit(x_train, y_train, regressor__epochs=100, regressor__validation_split=0.11, regressor__callbacks=early_stop)

    Train_score = R2_scorer(ANN, x_train , y_train)
    Test_score = R2_scorer(ANN, x_test, y_test)

    result = permutation_importance(ANN, x_test, y_test, scoring=R2_scorer, n_repeats=10)
    importances_bis = result.importances_mean

    df_entry1 = importances_bis.tolist()

    df_entry1.append(Train_score)
    df_entry1.append(Test_score)

    df1.loc[l] = df_entry1

# prints the pandas dataframe to csv
df1.to_csv("Results ANN yields Nguyen et al.csv")

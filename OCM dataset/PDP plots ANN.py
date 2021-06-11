import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import preprocessing
from sklearn.metrics import r2_score
from tensorflow.keras import layers
from sklearn.compose import ColumnTransformer
from sklearn.inspection import partial_dependence, plot_partial_dependence
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scikeras.wrappers import KerasRegressor

# read data
data = pd.read_csv("OCM-NguyenEtAl.csv")

# convert M1, M2 and M3 into numeric values, artefact of me working with multiple algorithms but superfluous here
le = preprocessing.LabelEncoder()
data["M1"] = le.fit_transform(data["M1"])
data["M2"] = le.fit_transform(data["M2"])
data["M3"] = le.fit_transform(data["M3"])

# define possible labels, features and dropped features
predict = ["CH4_conv", "C2y", "C2H6y", "C2H4y", "COy", "CO2y"]
categorical_columns = ["M1", "M2", "M3", "Support_ID"]
numerical_columns = ["M2_mol", "M3_mol", "Temp", "Ar_flow", "CH4_flow", "O2_flow", "M1_mol", "CT"]
features = numerical_columns + categorical_columns
dropped = ["CH4_conv", "C2y", "C2H6y", "C2H4y", "COy", "CO2y", "C2s", "C2H6s", "C2H4s", "COs", "CO2s", "Name",
           "M1_atom_number", "M2_atom_number", "M3_atom_number", "Support", "CH4/O2", "Total_flow", "M1_mol%",
           "M2_mol%", "M3_mol%"]

# calculate M1 mol as it is missing from the original dataset
data["M2_mol%"] = data["M2_mol%"] + 1e-7 # to avoid divide by zero errors
data["M1_mol"] = data["M1_mol%"] * (data["M2_mol"] + data["M3_mol"]) / (data["M2_mol%"] + data["M3_mol%"])

# chooses C2 yield as label
voorspel = predict[1]  # choose yield to predict

# making arrays of the features and label, standardizing label
X = data[features]
y = np.array(data[voorspel])
y = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
y = np.ravel(y.reshape(-1, 1))

# feature preprocessing method
categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
numerical_pipe = Pipeline([
    ('imputer', preprocessing.StandardScaler())
])

preprocessing = ColumnTransformer(
    [('cat', categorical_encoder, categorical_columns),
     ('num', numerical_pipe, numerical_columns)])

# early stopping criterion for the neural net
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# calculates R² regression score
def R2_scorer(ANN, x_test, y_test):
    y_pred = ANN.predict(x_test)
    score = r2_score(y_test,y_pred)
    return score

# constructs a neural net with keras sequential
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

# split data into test and validation + training data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# scikeras wrapper object to make the PDP package function with the keras neural net
kr = KerasRegressor(build_fn=build_model(), verbose=1, epochs=100)

ANN = Pipeline([
        ('preprocess', preprocessing),
        ('regressor', kr)
    ])


# fit the neural net to the data
ANN.fit(x_train, y_train, regressor__validation_split=0.11, regressor__callbacks=early_stop)

# show training and testing scores
Train_score = R2_scorer(ANN, x_train , y_train)
Test_score = R2_scorer(ANN, x_test, y_test)
print(Train_score)
print(Test_score)

# generates partial dependence plots
fig, ax = plt.subplots(figsize=(12, 6))

ax.set_title("Partial dependence plots for an ANN")
display = plot_partial_dependence(
    ANN, x_train, [2, 0, 5], ax=ax
)

plt.show()




import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.inspection import permutation_importance

# reads one of the generated synthetic datasets
data = pd.read_csv('Synthetic dataset orthogonal features.csv') #modify this text according to the dataset you want to analyze

# defining label and features
predict = 'Yield'
features = ['Metal_conc', 'Acid_conc', 'Gamma', 'Epsilon']

# make feature and label arrays
X = np.array(data.drop(predict, 1))
y = np.array(data[predict])

# standardizing the feature values
X = preprocessing.StandardScaler().fit_transform(X)
y = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
y = np.ravel(y.reshape(-1, 1))

# makes a pandas dataframe that'll contain the feature importances
columns = [feature for feature in features]
columns.append('Train score')
columns.append('Test score')

df_impurity = pd.DataFrame(columns=columns)
df_permutation = pd.DataFrame(columns=columns)

# loop that constructs models for different dataset splits, chooses the best model for a given dataset split and saves the respective feature importances to the dataframe
number_of_splits = 10
random_states = [0, 42, 10, 5]
for k in range(number_of_splits):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    x_train, x_validate, y_train, y_validate = sklearn.model_selection.train_test_split(x_train, y_train,
                                                                                        test_size=0.11)
    Train_score = 0
    Test_score = 0
    Validation_score = 0

    for random_state in random_states:
        forest = RandomForestRegressor(n_estimators=20, random_state=random_state)
        forest.fit(x_train, y_train)

        if (forest.score(x_validate, y_validate) > Validation_score):
            Train_score = forest.score(x_train, y_train)
            Test_score = forest.score(x_test, y_test)
            Validation_score = forest.score(x_validate, y_validate)

            importances = forest.feature_importances_.tolist()

            result = permutation_importance(forest, x_train, y_train, n_repeats=10)
            importances_bis = result.importances_mean.tolist()

    importances.append(Train_score)
    importances.append(Test_score)
    importances_bis.append(Train_score)
    importances_bis.append(Test_score)

    df_impurity.loc[k] = importances
    df_permutation.loc[k] = importances_bis

# printing the dataframes that contain the feature importances
df_impurity.to_csv('Impurity based feature importances.csv')
df_permutation.to_csv('Permutation based feature importances.csv')

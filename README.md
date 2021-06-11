# Recognition of key catalyst properties
To determine the feature importances for any of the databases that were used in the work, one simply needs to download the database or the database generation script and a python file corresponding to one of the supervised algorithms. The database generation script prints a synthetic database in csv form. Running a supervised algorithm python file will produce a csv file that contains the feature importances for ten different splits of the dataset. Pycharm was used as IDE to run the scripts.

## Generating synthetic datasets
Generating different synthetic datasets is done by modifying the for loop in the dataset generation script. Examples of this can be found in the appendices of the thesis. Adding random error to feature values does not require the generation of new datasets. One can simply multiply feature values by a certain factor in an already generated dataset.

## Dependencies

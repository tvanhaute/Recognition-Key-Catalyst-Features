# Recognition of key catalyst properties
To determine the feature importances for any of the databases that were used in the work, one simply needs to download the database or the database generation script and a python file corresponding to one of the supervised algorithms. The database generation script prints a synthetic database in csv form. Running a supervised algorithm python file will produce a csv file that contains the feature importances for ten different splits of the dataset. Pycharm was used as IDE to run the scripts. This IDE also contains plugins that aid in visualizing csv files. Make sure that the database and python file are in the same folder. An example for generating a partial dependence plot can also be found in the OCM dataset folder.

The OCM database that was used in the work was constructed by Nguyen et al. (https://doi.org/10.1021/acscatal.9b04293)

## Generating synthetic datasets
Generating different synthetic datasets is done by modifying the for loop in the dataset generation script. Examples of this can be found in the appendices of the thesis. Adding random error to feature values does not require the generation of new datasets. One can simply multiply feature values by a certain factor in an already generated dataset.

## Dependencies
* pandas 1.1.4
* scikit-learn 0.24.2
* numpy 1.19.4
* tensorflow 2.4.1
* scikeras 0.3.1
* matplotlib 3.3.3

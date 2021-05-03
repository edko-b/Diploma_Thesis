# -*- coding: utf-8 -*-

"""
Loads the first dataset and performs the first method.

sql_connection module is needed to import data.
"""

from sql_connection import engine as engine  # This module is not included in repo, to keep data private.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#from matplotlib import colors

# Loads a dataframe used to map the rat strain to rat ID.
strain_map = pd.read_sql(
        "SELECT `id`, `strain` FROM `ratExperiment`",
        con=engine
    )

# Define a dictionary that maps rat strains to a colour. The color is kept consistent throughot the whole work.
colors_map = {'SHR': '#F5793A', 'WT': '#A95AA1', 'HanSD': '#0F2080', 'TGR': '#85C0F9'}

# Define dictionary encoder. Most of the methods encode strings automaticaly, but there are exeptions.
encoder = {'SHR': '0', 'WT': '1', 'HanSD': '2', 'TGR': '3'}

# Names of the fields of the dataframe. Used for properly labeling axes and names of the plots.
column_title = {'pulsePressure': 'Pulse Pressure (mmHg)',
                'diastolicBP': 'Diastolic BP (mmHg)',
                'meanBP': 'Blood Pressure (mmHg)',
                'systolicBP': 'Systolic BP (mmHg)',
                'heartRate': 'Heart Rate (BPM)',
                'activity': 'Activity (Movement Per Minute)'}


def main():
    """Wrap  the main process separate."""
    from control_select import control_df

    def plot_day_night_scatter(column_name, rats_avgs):
        """
        Create a scatter plot with dark phase averages on X axis and light phase averages on the Y axis.

        Parameters
        ----------
        column_name : string
            Selct which column to plot.
        rats_avgs : pandas.dataframe
            Dataframe of aggregate data.


        """
        plt.figure()
        rats_avgs = rats_avgs.join(strain_map.set_index('id'))
        plt.scatter(column_name + 'dark',
                    column_name + 'light',
                    data=rats_avgs,
                    c=rats_avgs['strain'].map(colors_map))
        plt.title(column_title[column_name])
        plt.xlabel('Dark Phase')
        plt.ylabel('Light Phase')
        plt.show()
        plt.savefig('rats_avgs_' + str(column_name) + '.png',
                    dpi=300,
                    transparent=True)

    def calc_avg_one_column(column_name, kresli=True):
        """
        Calculate averages separate for light and dark phase.

        Parameters
        ----------
        column_name : string
            Selct which column to use.
        kresli : boolean, optional
            If the scatter plot of that column should be plotted. The default is True.

        Returns
        -------
        rats_avgs : pandas.dataframe
            calculated average values per rat.

        """
        pomocna = control_df[['idRatExperiment', column_name, 'lightIntensity']]
        pivot_rats = pomocna.groupby(['idRatExperiment', 'lightIntensity']).mean()
        pivot_rats = pivot_rats.reset_index(level=[0, 1])
        X = pivot_rats[['idRatExperiment', column_name]][pivot_rats.lightIntensity == 0]
        Y = pivot_rats[['idRatExperiment', column_name]][pivot_rats.lightIntensity == 150]
        rats_avgs = X.set_index('idRatExperiment').join(Y.set_index('idRatExperiment'), lsuffix='dark', rsuffix='light')

        if kresli:
            plot_day_night_scatter(column_name, rats_avgs)

        return rats_avgs

    def calc_avg_one_column_multiple(kresli=True, *args):
        """
        Call calc_avg_one_column() function on multiple columns.

        Parameters
        ----------
        kresli : boolean, optional
            If the scatter plots of that columns should be plotted. The default is True.
        *args : string
            Names of the columns requested.

        Returns
        -------
        x : pandas.dataframe
            Rats with the corresponding average values. All the needed features are now prepared.

        """
        iterargs = iter(args)  # Goes through the multipe arguments (column names)
        x = calc_avg_one_column(next(iterargs), kresli)  # Calls the function to calculate the averages.
        for column_name in iterargs:
            x = x.join(calc_avg_one_column(column_name, kresli))

        x = x.join(strain_map.set_index('id'))

        return x

    table_avg = calc_avg_one_column_multiple(False,  # We don't want to plot the scatterplots now
                                                    'pulsePressure',
                                                    'diastolicBP',
                                                    'meanBP',
                                                    'systolicBP',
                                                    'heartRate',
                                                    'activity')

    # Transform the features from pandas.df into numpy.array
    X = np.array(table_avg[['pulsePressuredark', 'pulsePressurelight',
                            'diastolicBPdark', 'diastolicBPlight',
                            'meanBPdark', 'meanBPlight',
                            'systolicBPdark', 'systolicBPlight',
                            'heartRatedark', 'heartRatelight',
                            'activitydark', 'activitylight']])

    # Prepare the labels, transform into numpy.array
    y = np.ravel(np.array(table_avg[['strain']]))

    # Split the dataset into the training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    # Scale(standardize) the features. Transform both of the sets, but train only on training set.
    # We treat the test set as not seen before the training.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Prepare the grid of parameters for grid-search. We want to find the best C parameter.
    # Kernel is left from when we tried other kernels, before deciding to keep it simple.
    param_grid = {
    'C': [2**(2*i-1) for i in range(-2,9)],
    'kernel': ['linear']
    }

    # Perform the grid search.
    clf = GridSearchCV(SVC(),
                       param_grid,
                       scoring='f1_micro',
                       n_jobs = -1)
    clf.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % clf.best_score_)
    print(clf.best_params_)

    # Use the model trained with the best value of hyperparameter C, to validate on test set.
    pred = clf.best_estimator_.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))




# =============================================================================
#  # Plot exemplary SVM
#  # Exemplary figure to show SVM in 2D
#
#     y = np.ravel(np.array(table_avg[['strain']]))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#
#     clf = svm.SVC(kernel='linear')
#     clf.fit(X_train[:, [5,9]], y_train)
#     y_pred = clf.predict(X_test[:,[5,9]])
#     print(confusion_matrix(y_test,y_pred))
#     print(classification_report(y_test,y_pred))
#
#     fig = plt.figure()
#     ax = plt.gca()
#     x_min, x_max = X_test[:,5].min() - 1, X_test[:,5].max() + 1
#     y_min, y_max = X_test[:,9].min() - 1, X_test[:,9].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/1000), np.arange(y_min, y_max, (y_max-y_min)/1000))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     cmap = colors.ListedColormap(['#F5793A','#A95AA1','#0F2080','#85C0F9'])
#     bounds = [0, 0.9, 1.9, 2.9, 3.9]
#     norm = colors.BoundaryNorm(bounds, cmap.N)
#     ax.pcolor(xx, yy, (np.vectorize(encoder.get)(Z)).astype(int), cmap=cmap, norm=norm)
#     ax.scatter(X_test[:,5], X_test[:,9], c=(np.vectorize(encoder.get)(y_test)).astype(int), cmap=cmap, norm=norm, s=40, edgecolors='k')
#     ax.xaxis.label.set_fontsize(10)
#     ax.yaxis.label.set_fontsize(10)
#     ax.set_xlabel('Blood Pressure (mmHg)')
#     ax.set_ylabel('Heart Rate (BPM)')
#     #plt.savefig('svm_heartRate.png', dpi=300, transparent=True)
# =============================================================================


# Wrapper for executing the main script.
# We do this to ensure that the long process of training the ML model is not executed when importing the strain map.
if __name__ == "__main__":
         main()

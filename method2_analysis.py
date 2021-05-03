# -*- coding: utf-8 -*-
"""
Prepare features and train SVM
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import copy
import seaborn as sns
import numpy as np
from Cluster_avg import colors_mapa
from tkinter.filedialog import askopenfilenames
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
sns.set_style("whitegrid")


def load_dict_dialog():
    """
    Dialog window opens up.
    Need to manualy select the files prepared by method2_prepare_dataset.
    Creates Large dictionary that includes all the data.

    Returns
    -------
    Files_Dict: dictionary of dictionaries of dictionaries of dictionaries of pandas.dataframe
        The one large dictionary containing all the data.

    """
    temp = askopenfilenames()
    file_names = list(temp)

    # Prepare the object for loading.
    Files_Dict = {'SBP': {'HanSD': {0: {}, 1: {}, 2: {}},
                          'TGR': {0: {}, 1: {}, 2: {}},
                          'SHR': {0: {}, 1: {}, 2: {}, 3: {}},
                          'WT': {0: {}, 1: {}, 2: {}, 3: {}}},
                  'IBI': {'HanSD': {0: {}, 1: {}, 2: {}},
                          'TGR': {0: {}, 1: {}, 2: {}},
                          'SHR': {0: {}, 1: {}, 2: {}, 3: {}},
                          'WT': {0: {}, 1: {}, 2: {}, 3: {}}}}

    # Regex used to find keys to populate the dictionaries
    for filename in tqdm(file_names):
        ID_rat = re.search(r"_(\d+)_", filename).group(1)
        strain = re.search(r"_(HanSD|TGR|SHR|WT)_", filename).group(1)
        field = re.search(r"_(SBP|IBI)", filename).group(1)
        alt = re.search(r"_alt_(\d)", filename).group(1)
        with open(filename, "rb") as fin:
            Files_Dict[field][strain][int(alt)][ID_rat] = pickle.load(fin)
    return Files_Dict


def create_two_df(main_dict_original):
    """
    Dismantle the dict of dict of... into two large datasets. One for blood pressure one for interbeat intervals

    Parameters
    ----------
    main_dict_original : Dict of..
        The large loaded dictionary.

    Returns
    -------
    main_dict : Dictionary

    """
    main_dict = copy.deepcopy(main_dict_original)
    for field, v in main_dict.items():  # through IBI/SBP
        for strain, u in v.items():  # throug strain
            for alt, w in u.items():
                for ID_rat, x in w.items():
                    main_dict[field][strain][alt][ID_rat] = pd.concat(x).reset_index(level=[0, 1]).drop(columns=['level_1']).rename(columns={'level_0': 'hour'})
                main_dict[field][strain][alt] = pd.concat(w).reset_index(level=[0, 1]).drop(columns=['level_1']).rename(columns={'level_0': 'ID_rat'})
            main_dict[field][strain] = pd.concat(u).reset_index(level=[0, 1]).drop(columns=['level_1']).rename(columns={'level_0': 'alt'})
        main_dict[field] = pd.concat(v).reset_index(level=[0, 1]).drop(columns=['level_1']).rename(columns={'level_0': 'strain'})
    main_dict['SBP']['hour'] = main_dict['SBP']['hour'].astype('int')
    main_dict['IBI']['hour'] = main_dict['IBI']['hour'].astype('int')
    return main_dict


def nakresli_box_ploty(main_dict_df):
    """
    Create two boxplots for each of the characteristics.

    Parameters
    ----------
    main_dict_df : dictionary of dataframe

    Returns
    -------
    None.

    """
    # Load the light/dark cycle info for each rat.
    with open('rezim', "rb") as fin:
        rezim = pickle.load(fin)

    # With modulo find which hour of the day the recording is
    main_dict_df['IBI']['order'] = main_dict_df['IBI'].apply(lambda x: (x.hour-int(rezim[x.ID_rat])) % 24, axis=1)
    main_dict_df['SBP']['order'] = main_dict_df['SBP'].apply(lambda x: (x.hour-int(rezim[x.ID_rat])) % 24, axis=1)

    fig, ax = plt.subplots()
    # plt.ylim(0.1, 0.4)
    g = sns.boxplot(x='order', y='IBI', hue='strain', data=main_dict_df['IBI'],
                    palette=colors_mapa, showfliers=False
                    ).set(xlabel='Hour', ylabel='Inter-Beat-Interval (s)')
    ax.legend_.remove()
    fig, ax = plt.subplots()
    g = sns.boxplot(x='order', y='SBP', hue='strain', data=main_dict_df['SBP'],
                    palette=colors_mapa, showfliers=False
                    ).set(xlabel='Hour', ylabel='Systolic Blood Pressure (mmHg)')
    ax.legend_.remove()


main_dict = load_dict_dialog()
main_dict_df = create_two_df(main_dict)
# nakresli_box_ploty(main_dict_df)


# CCalculate the moments
IBI_aggregate = main_dict_df['IBI'].groupby(['ID_rat', 'alt', 'hour', 'strain']).agg({'IBI': ['describe']})
SBP_aggregate = main_dict_df['SBP'].groupby(['ID_rat', 'alt', 'hour', 'strain']).agg({'SBP': ['describe']})

# Reset the indexes, so that the categorical values are in columns
IBI_aggregate = IBI_aggregate.reset_index(level=[0, 1, 2, 3])
SBP_aggregate = SBP_aggregate.reset_index(level=[0, 1, 2, 3])

# Join the two dataframes
main_df = IBI_aggregate.merge(SBP_aggregate, on=['ID_rat', 'alt', 'hour', 'strain'])

# =============================================================================
# # Create dummy variable light NOT USED IN FINAL
# with open('rezim', "rb") as fin:
#     rezim = pickle.load(fin)
# main_df['light'] = main_df.apply(lambda x: (x.hour-int(rezim[str(int((x.ID_rat)))]))%24, axis=1) < 12
# =============================================================================

main_df = main_df.drop(['count', 'min', 'max'], axis=1, level=2)


# Apply the same machine learning approach as in other methods
X = main_df[['IBI', 'SBP']].to_numpy()
y = np.ravel(main_df[['strain']].to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {
    'C': [2**(2*i-1) for i in range(-2, 9)],
    'kernel': ['linear']
    }

clf = GridSearchCV(SVC(), param_grid, scoring='f1_micro', n_jobs=-1)
clf.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % clf.best_score_)
print(clf.best_params_)

clf.best_estimator_.predict(X_train)
pred = clf.best_estimator_.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

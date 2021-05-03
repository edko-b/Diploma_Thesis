# -*- coding: utf-8 -*-
"""
Extract the features of the triangles and train the model.
"""

from tkinter.filedialog import askopenfilenames
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import pickle
import numpy as np
import pandas as pd
import math
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from matplotlib import colors


def find_polar_r(dataframe):
    """
    Transform into polar, find r.

    Parameters
    ----------
    dataframe : pandas.dataframe
        Dataframe of 1 rat.

    Returns
    -------
    dataframe : pandas.dataframe
        the same dataframe with the calculated r values appended.

    """
    dataframe['r'] = dataframe.apply(lambda x: math.hypot(x.u,x.v), axis=1)
    return dataframe


def find_polar_phi(dataframe):
    """
    Transform into polar, find angle.

    Parameters
    ----------
    dataframe : pandas.dataframe
        Dataframe of 1 rat.

    Returns
    -------
    dataframe : pandas.dataframe
        the same dataframe with the calculated angle values appended.

    """
    dataframe['phi'] = dataframe.apply(lambda x: math.atan2(x.v, x.u), axis=1)  # !!! v and u in unusual order
    return dataframe


def load_optimal_values():
    """
    Load the optimal values found for each recording.
    Open dialog window and select the files.

    Returns
    -------
    Files_Dict : Dictionary
        Dictionary assigning the optimal values to the recordings.

    """
    temp = askopenfilenames()
    file_names = list(temp)
    Files_Dict = {42: {0: {}, 1: {}, 2: {}},
                  52: {0: {}, 1: {}, 2: {}},
                  74: {0: {}, 1: {}, 2: {}, 3: {}},
                  146: {0: {}, 1: {}, 2: {}, 3: {}}}
    for filename in tqdm(file_names):
        ID_rat = re.search(r"id_(\d+)_", filename).group(1)
        alt = re.search(r"_alt_(\d)", filename).group(1)
        with open(filename, "rb") as fin:
            Files_Dict[int(ID_rat)][int(alt)] = pickle.load(fin)
    return Files_Dict


def aux_bin_middle(bin_edges):
    """
    Find centrpoints of the bins.
    Auxiliary function.

    Parameters
    ----------
    bin_edges : array of dtype float
        array specifying the edges.

    Returns
    -------
    bin_centers : numpy.array
        array of bin centers.

    """
    bin_centers = []
    for i in range(len(bin_edges) -  1):
        center = bin_edges[i] + (float(bin_edges[i + 1]) - float(bin_edges[i]))/2.
        bin_centers.append(center)
    return np.array(bin_centers)


def evaluate_stats(df_rat, kresli=False):
    """
    Create the features for ML algorithm.

    Parameters
    ----------
    df_rat : pandas.dataframe
        Dataframe of 1 recording.
    kresli : boolean, optional
        If the plot of polar transform should be plotted.. The default is False.

    Returns
    -------
    phi_vertex : float
        Attractor rotation.
    r_circumscribed : float
        Attractor size circumscribed.
    vertex_width : float
        Width in vertex.
    vertex_count : integer
        Count in vertex.
    phi_edge : float
        Attractor rotation defined by the edge.
    r_inscribed : float
        Attractor size inscribed.
    edge_width : float
        Width in edge.
    edge_count : integer
        Count in edge.

    """
    bin_mean, bin_edges, binnumber = stats.binned_statistic(x=df_rat.phi, values=df_rat.r, bins=100, statistic='mean')
    bin_std, bin_edges, binnumber = stats.binned_statistic(x=df_rat.phi, values=df_rat.r, bins=100, statistic='std')
    bin_hist, bin_edges, binnumber = stats.binned_statistic(x=df_rat.phi, values=df_rat.r, bins=100, statistic='count')

    if kresli:
        f1 = sns.jointplot(data=df_rat, x="phi", y="r", color='#85C0F9')
        f1.ax_joint.hlines(bin_mean, bin_edges[:-1], bin_edges[1:], colors='#A95AA1', lw=2, label='binned statistic of data')
        f1.ax_joint.vlines(aux_bin_middle(bin_edges), bin_mean - bin_std, bin_mean + bin_std, colors='#0F2080')
        plt.figure()
        plt.axes(projection='polar')
        plt.polar(df_rat.phi, df_rat.r)

    bin_mids = aux_bin_middle(bin_edges)

    mask_limit = ((bin_mids < 0) & (bin_mids > -2))  # To make sure that we find the first extreme value
    max_index = bin_mean[mask_limit].argmax()
    phi_vertex = bin_mids[mask_limit][max_index]
    r_circumscribed = bin_mean[mask_limit][max_index]
    vertex_width = bin_std[mask_limit][max_index]
    vertex_count = bin_hist[mask_limit][max_index]

    mask_limit_moved = ((bin_mids < phi_vertex + 2/3*math.pi) & (bin_mids > phi_vertex))
    min_index = bin_mean[mask_limit_moved].argmin()
    phi_edge = bin_mids[mask_limit_moved][min_index]
    r_inscribed = bin_mean[mask_limit_moved][min_index]
    edge_width = bin_std[mask_limit_moved][min_index]
    edge_count = bin_hist[mask_limit_moved][min_index]

    return phi_vertex, r_circumscribed, vertex_width, vertex_count, phi_edge, r_inscribed, edge_width, edge_count


X_dict = {42: {0: {}, 1: {}, 2: {}},
          52: {0: {}, 1: {}, 2: {}},
          74: {0: {}, 1: {}, 2: {}, 3: {}},
          146: {0: {}, 1: {}, 2: {}, 3: {}}}

optimal_values = load_optimal_values()


###############################################################################
# Here we load the files prepared in method3_prepare_dataset.py and do the transformations.
# Some loading function needs to be imported form method3_find_optimal.
# Data are then transformed into polar by functions defined above.
# This takes several minute, so we saved the results after the first time.
# We commented the block and each time only load the already transformed data from file 'prepared_dict_5min'.
###############################################################################

# =============================================================================
# from method3_find_optimal import *
# dict_rats = load_dict_dialog()
#
#
# for ID_rat,v in dict_rats.items():
#         for alt,u in tqdm(v.items()):
#             for hour,w in u.items():
#                 X_dict[ID_rat][alt][hour] = create_new_variables(w, [t[1] for t in optimal_values[ID_rat][alt] if t[0]==hour][0])
#                 X_dict[ID_rat][alt][hour] = find_polar_r(X_dict[ID_rat][alt][hour])
#                 X_dict[ID_rat][alt][hour] = find_polar_phi(X_dict[ID_rat][alt][hour])
# =============================================================================

# =============================================================================
# with open('prepared_dict_5min', "wb") as fin:
#     pickle.dump(X_dict, fin)
# =============================================================================

with open('C:/Users/ide23/Documents/Diplo/prepared_dict_5min', "rb") as fin:
    X_dict = pickle.load(fin)


id_strain_map = {
    42:'HanSD',
    52:'TGR',
    74:'SHR',
    146:'WT'
    }

X_df = pd.DataFrame(columns=['ID_rat', 'alt', 'hour', 'phi_vertex',
                             'r_circumscribed', 'vertex_width',
                             'vertex_edge_count_ration', 'phi_edge',
                             'r_inscribed', 'edge_width', 'tau'])
for ID_rat, v in X_dict.items():
    for alt, u in tqdm(v.items()):
        for hour, w in u.items():
            stat = list(evaluate_stats(w))
            novy_riadok = {
                'ID_rat': int(ID_rat),
                'alt': int(alt),
                'hour': int(hour),
                'phi_vertex': stat[0],
                'r_circumscribed': stat[1],
                'vertex_width': stat[2],
                'vertex_edge_count_ration': stat[3]/stat[7],
                'phi_edge': stat[4],
                'r_inscribed': stat[5],
                'edge_width': stat[6],
                'tau': [t[1] for t in optimal_values[ID_rat][alt] if t[0] == hour][0]
                }
            X_df = X_df.append(novy_riadok, ignore_index=True)


X_df['strain'] = X_df['ID_rat'].map(id_strain_map)

X = X_df[['phi_vertex', 'r_circumscribed', 'vertex_width',
          'vertex_edge_count_ration', 'phi_edge',
          'r_inscribed', 'edge_width', 'tau']].to_numpy()
y = np.ravel(X_df[['strain']].to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {
    'C': [2**(2*i-1) for i in range(-2, 9)],
    'kernel': ['linear']
    }

clf = GridSearchCV(SVC(),
                   param_grid,
                   scoring='f1_micro',
                   n_jobs = -1)
clf.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % clf.best_score_)
print(clf.best_params_)

clf.best_estimator_.predict(X_train)
pred = clf.best_estimator_.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


def f_importances(coef, names):
    """
    Evaluate feature importance of features with linear kernel.

    Parameters
    ----------
    coef : list
        List of coef. values.
    names : list
        List of coef. names.

    Returns
    -------
    None.

    """
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


features_names = ['phi_vertex', 'r_circumscribed', 'vertex_width',
                  'vertex_edge_count_ration', 'phi_edge',
                  'r_inscribed', 'edge_width', 'tau']
f_importances(clf.best_estimator_.coef_[0], features_names)


def vykresli_animaciu(dict_potkan, optimal_values):
    from method3_find_optimal import create_histograms
    from matplotlib import animation
    fig = plt.figure(figsize=[20, 10], frameon=False)

    cmap = colors.LinearSegmentedColormap.from_list("", ["#d5f2f5", "#002e47", "#eb6d19"])

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ims = []
    for hour in range(0, 24):
        H1 = create_histograms(dict_potkan[42][0][str(hour)])[0]
        H2 = create_histograms(dict_potkan[52][0][str(hour)])[0]
        H3 = create_histograms(dict_potkan[74][0][str(hour)])[0]
        H4 = create_histograms(dict_potkan[146][0][str(hour)])[0]

        im1 = ax1.imshow(np.sqrt(H1), cmap=cmap,
                         extent=[-H1.shape[1]/2., H1.shape[1]/2., -H1.shape[0]/2., H1.shape[0]/2.],
                         animated=True,)
        im2 = ax2.imshow(np.sqrt(H2), cmap=cmap,
                         extent=[-H1.shape[1]/2., H1.shape[1]/2., -H1.shape[0]/2., H1.shape[0]/2.],
                         animated=True)
        im3 = ax3.imshow(np.sqrt(H3), cmap=cmap,
                         extent=[-H1.shape[1]/2., H1.shape[1]/2., -H1.shape[0]/2., H1.shape[0]/2.],
                         animated=True)
        im4 = ax4.imshow(np.sqrt(H4), cmap=cmap,
                         extent=[-H1.shape[1]/2., H1.shape[1]/2., -H1.shape[0]/2., H1.shape[0]/2.],
                         animated=True)

        cas1 = ax1.text(0.5, 0.90, 'hour: ' + str(hour), ha="center", va="bottom",
                        fontsize="large", transform = ax1.transAxes)

        tau1 = ax1.text(0.5, 0.10, 'Tau: ' + str([t[1] for t in optimal_values[42][0] if t[0] == str(hour)][0]),
                        ha="center", va="bottom", fontsize="large", transform=ax1.transAxes)

        cas2 = ax2.text(0.5, 0.90, 'hour: ' + str(hour), ha="center", va="bottom",
                        fontsize="large", transform=ax2.transAxes)

        tau2 = ax2.text(0.5, 0.10, 'Tau: ' + str([t[1] for t in optimal_values[52][0] if t[0] == str(hour)][0]),
                        ha="center", va="bottom", fontsize="large", transform=ax2.transAxes)

        cas3 = ax3.text(0.5, 0.90, 'hour: ' + str(hour), ha="center", va="bottom",
                        fontsize="large", transform=ax3.transAxes)

        tau3 = ax3.text(0.5, 0.10, 'Tau: ' + str([t[1] for t in optimal_values[74][0] if t[0] == str(hour)][0]),
                        ha="center", va="bottom", fontsize="large", transform=ax3.transAxes)

        cas4 = ax4.text(0.5, 0.90, 'hour: ' + str(hour), ha="center", va="bottom",
                        fontsize="large", transform=ax4.transAxes)

        tau4 = ax4.text(0.5, 0.10, 'Tau: ' + str([t[1] for t in optimal_values[146][0] if t[0] == str(hour)][0]),
                        ha="center", va="bottom", fontsize="large", transform=ax4.transAxes)
        ims.append([im1, im2, im3, im4, cas1, tau1, cas2, tau2, cas3, tau3, cas4, tau4])
    ani = animation.ArtistAnimation(fig, ims, interval=5000, blit=True,
                                    repeat_delay=100)


    writergif = animation.PillowWriter(fps=2)
    ani.save("trojuholnik_porovnanie_upgrade_vsetky.gif",
             writer=writergif, dpi = 300)
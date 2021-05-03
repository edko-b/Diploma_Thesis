# -*- coding: utf-8 -*-
"""
Load the dictionaries prepared by method3_prepare_dataset.
Find optimal triangles.
Save optimal vlaues.
"""

from tkinter.filedialog import askopenfilenames
from tqdm import tqdm
import re
import pickle
import copy
import numpy as np
import pandas as pd
import multiprocessing

pd.options.mode.chained_assignment = None  # Suppres some warnings


def load_dict_dialog():
    """
    Dialog window opens up.
    Need to manualy select the files prepared by method3_prepare_dataset.
    Creates Large dictionary that includes all the data

    Returns
    -------
    Files_Dict : Dictionary of dictionaries of dictionaries of dictionaries of dataframes
        DESCRIPTION.

    """
    temp = askopenfilenames()
    file_names = list(temp)
    # Prepares the dict of dict of dict for loading
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


def create_lags(dataframe_rat, tau, number_lags=2):
    """
    Create lagged values of the signal.

    Parameters
    ----------
    dataframe_rat : pandas.dataframe
        Dataframe of single recording.
    tau : integer
        Number by which should the new signal be lagged.
    number_lags : int, optional
        How many new variables lags. The default is 2. Shouldn't put any other value.

    Returns
    -------
    dataframe_rat : pandas.dataframe
        The inputted dataframe with the new lagged variables added.

    """

    for lag in range(1, number_lags + 1):
        dataframe_rat['lag_' + str(lag)] = dataframe_rat.ABP.shift(tau * lag)
    dataframe_rat = dataframe_rat.dropna(0)
    return dataframe_rat

def transform_into_u_v(dataframe_rat):
    """
    Transfor the signal and its lags into new features

    Parameters
    ----------
    dataframe_rat : pandas.dataframe
        The dataframe with lags already created.

    Returns
    -------
    dataframe_rat : pandas.dataframe
        The inputted dataframe with the new features u and v added.

    """

    # Try except to handle some error values
    try:
        dataframe_rat['u'] = (1 / np.sqrt(6)) * (dataframe_rat.ABP - 2 * dataframe_rat.lag_1 + dataframe_rat.lag_2)
        dataframe_rat['v'] = (1 / np.sqrt(2)) * (dataframe_rat.ABP - dataframe_rat.lag_2)
    except TypeError:
        dataframe_rat = dataframe_rat.apply(pd.to_numeric, errors='coerce')
        dataframe_rat['u'] = (1 / np.sqrt(6)) * (dataframe_rat.ABP - 2 * dataframe_rat.lag_1 + dataframe_rat.lag_2)
        dataframe_rat['v'] = (1 / np.sqrt(2)) * (dataframe_rat.ABP - dataframe_rat.lag_2)

        # I considered detecting outliers in the u,v space. Would probably need to use proper 2Dim method.
   # dataframe_rat = dataframe_rat[np.abs(dataframe_rat.u-dataframe_rat.u.mean()) <= (3*dataframe_rat.u.std())]
   # dataframe_rat = dataframe_rat[np.abs(dataframe_rat.v-dataframe_rat.v.mean()) <= (3*dataframe_rat.v.std())]

    return dataframe_rat


def create_new_variables(rat_raw, tau):
    """
    Create the u and v variables needed to create attractor.
    Steps:
        Handle outliers.
        Call the function to create lagged variables.
        Transform the signals.


    Parameters
    ----------
    rat_raw : pandas.dataframe
        Original dataframe of one rat.
    tau : int
        Time delay.

    Returns
    -------
    rat_one : pandas.dataframe
        New dataframe copy with new variables u and v added.

    """
    rat_one = copy.deepcopy(rat_raw)  # Creates copy, so that the original dataframe is not rewritten.
    r = rat_one.rolling(window=50, center=True)
    mps_up, mps_low = r.mean() + 3 * r.std(), r.mean() - 3 * r.std()
    # Fortunately using at least one NaN vlaue in the transformations results also in Nan
    # So the outlier status is heredited to all the u, v values that contain the outlier datapoint
    rat_one.loc[~rat_one['ABP'].between(mps_low.ABP, mps_up.ABP), 'ABP'] = np.NaN
    rat_one = create_lags(rat_one, tau=tau)
    rat_one = transform_into_u_v(rat_one)

    return rat_one


def create_histograms(rat_one, bin_width=0.5):
    """
    Create 3 histograms of the u,v variables in 2Dim.

    Parameters
    ----------
    rat_one : pandas.dataframe
        A dataframe with the new variables u and v added.
    bin_width : float, optional
        The width of one bin of histogram. The default is 0.5.

    Returns
    -------
    H1 : numpy.array
        The first histogram.
    H2 : numpy.array
        The second histogram. Histogram of the 2Dim image rotated by 120 degrees.
    H3 : numpy.array
        The third histogram. Histogram of the 2Dim image rotated by 240 degrees.

    """

    # Define the bins.
    custom_bins = np.arange(rat_one[['u', 'v']].min().min(), rat_one[['u', 'v']].max().max()+bin_width, bin_width)

    # Cast to numpy vectors.
    u = rat_one['u'].to_numpy()
    v = rat_one['v'].to_numpy()

    # Prepare the rotation matrix.
    theta = np.radians(120)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # Concatenate the vectors into a matrix.
    points = np.column_stack((u,v))

    # Create rotated images.
    rotated1 = points @ R.T
    rotated2 = rotated1 @ R.T

    # Create histograms of the images.
    H1, xedges, yedges = np.histogram2d(v, u, bins=[custom_bins, custom_bins])
    H2, xedges, yedges = np.histogram2d(rotated1[:, 1], rotated1[:, 0], bins=[custom_bins, custom_bins])
    H3, xedges, yedges = np.histogram2d(rotated2[:, 1], rotated1[:, 0], bins=[custom_bins, custom_bins])
    return H1,H2,H3


def eval_symmetry_measure(H1, H2, H3):
    """
    Calculate the symmetry measure

    Parameters
    ----------
    H1 : numpy.array
        The first histogram.
    H2 : numpy.array
        The second histogram. Histogram of the 2Dim image rotated by 120 degrees.
    H3 : numpy.array
        The third histogram. Histogram of the 2Dim image rotated by 240 degrees.

    Returns
    -------
    train_error : float
        The symmetry measure.

    """
    # Combine the triangles
    comb_triangle = (H1+H2+H3)/3

    # Quantify the difference between the histograms.
    train_error = np.linalg.norm(comb_triangle-H1, 2)
    return train_error


def find_the_first_minimum(error_track):
    """
    Find the first minimum.
    The function of symmetry measure is divided into segments of length 5.
    We assume that it is decreasing in terms of these segments.

    Parameters
    ----------
    error_track : List
        List of symmetry measures per tau.

    Returns
    -------
    t : int
        Argmin. the optimal tau.
    actual_min : float
        minimal value of the symmetry measure.

    """
    i = 0
    found = False
    future_min = min(error_track[0:5])
    while not found:
        actual_min = future_min
        future_min = min(error_track[5*(i+1):5*(i+2)])
        if actual_min < future_min:
            found = True
            t = 5*i + error_track[5*i:5*(i+1)].index(actual_min) + 1
        i = i + 1
    return t, actual_min

def find_optimal_values(item, maxtau = 100):
    """
    Find the optimal tau and symmetry measure for given hour.
    Defines the proces for separate cores of the processor.
    Calls the functions above.


    Parameters
    ----------
    item : touple?
        (Hour : int, df_rat : pandas.dataframe).
    maxtau : integer, optional
        The maximal considered tau. The default is 100.

    Returns
    -------
    hour : integer
        The hour currently considered.
    tau : integer
        The optimal delay.
    value : float
        The minimal symmetry measure.

    """
    hour, df_rat = item
    error_track = []
    for t in range(1, maxtau): # prejdeme mozne tau
        rat_one = create_new_variables(df_rat, tau=t)
        H1, H2, H3 = create_histograms(rat_one[-(100000):], bin_width=0.5)
        chyba = eval_symmetry_measure(H1, H2, H3)
        error_track.append(chyba)
    tau, value = find_the_first_minimum(error_track)
    return hour, tau, value


#  Wrapped into the name=__main__ wrapper to secure smooth multiprocessing.
if __name__ == '__main__':
    large_dict = load_dict_dialog()

    for k, v in large_dict.items():
        for l, u in v.items():
            with multiprocessing.Pool(processes=8) as pool:
                vals = list(tqdm(pool.imap(find_optimal_values, list(u.items())), total=len(u.items())))
            # Save the optimal values
            with open('optimalne_id_' + str(k) + '_alt_' + str(l), "wb") as fin:
                pickle.dump(vals, fin)

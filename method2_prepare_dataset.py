# -*- coding: utf-8 -*-
"""
The data came nested in separately in folders.
To speed up their loading, we loaded them once,
connected into large dictionaries of dataframes,
saved as pickle files.
Next time, only the pickle files need to be quickly loaded.

"""

from tkinter.filedialog import askopenfilenames
import re
import pandas as pd
# import matplotlib.pyplot as plt
import pickle
import os
import natsort


def load_rat_save(file_names=None):
    """
    Load the data one by one.
    Assign to the correct place in the dict of dataframes.
    Save the final dictionaries

    Parameters
    ----------
    file_names : string, optional
        The paths to the files containing the data. The default is None.

    Returns
    -------
    None.

    """
    # If we dont specify filenames, file dialog is opened where we can select what files we want to load.
    if file_names is None:
        temp = askopenfilenames()
        file_names = list(temp)

    # Regex to find the info about the rat ID
    idcko = re.search(r"id(\d+)", file_names[0]).group(1)
    strain = re.search(r"(HanSD|TGR|SHR|WT)", file_names[0]).group(1)
    stlpec = re.search(r"(SBP|IBI)", file_names[0]).group(1)

    # The rats had their records taken multiple time per hour, this is to find which of the recordings it is.
    # We use it to create separate ALTERNATIVES of the recording
    minuta = {0: re.search(r"-M(\d+)", file_names[0]).group(1)}
    for i in [1, 2, 3]:
        if re.search(r"-M(\d+)", file_names[i]).group(1) != minuta[0]:
            minuta[i] = re.search(r"-M(\d+)", file_names[i]).group(1)

    hodina = re.search(r"_H(\d+)", file_names[0]).group(1)
    DataFrameDict = {}
    DataFrameDict_alt_1 = {}
    DataFrameDict_alt_2 = {}
    DataFrameDict_alt_3 = {}
    for filename in file_names:
        if re.search(r"-M(\d+)", filename).group(1) == minuta[0]:
            hodina = re.search(r"_H(\d+)", filename).group(1)
            DataFrameDict[hodina] = pd.read_csv(filename,
                                                header=None,
                                                names=['time', stlpec])

        if re.search(r"-M(\d+)", filename).group(1) == minuta[1]:
            hodina = re.search(r"_H(\d+)", filename).group(1)
            DataFrameDict_alt_1[hodina] = pd.read_csv(filename,
                                                      header=None,
                                                      names=['time', stlpec])

        if re.search(r"-M(\d+)", filename).group(1) == minuta[2]:
            hodina = re.search(r"_H(\d+)", filename).group(1)
            DataFrameDict_alt_2[hodina] = pd.read_csv(filename,
                                                      header=None,
                                                      names=['time', stlpec])
        try:
            if re.search(r"-M(\d+)", filename).group(1) == minuta[3]:
                hodina = re.search(r"_H(\d+)", filename).group(1)
                DataFrameDict_alt_3[hodina] = pd.read_csv(filename,
                                                          header=None,
                                                          names=['time', stlpec])
        except:
            pass

    with open('BTB_' + strain + '_id_' + idcko + '_' + stlpec + '_alt_0', "wb") as fin:
        pickle.dump(DataFrameDict, fin)

    with open('BTB_' + strain + '_id_' + idcko + '_' + stlpec + '_alt_1', "wb") as fin:
        pickle.dump(DataFrameDict_alt_1, fin)

    with open('BTB_' + strain + '_id_' + idcko + '_' + stlpec + '_alt_2', "wb") as fin:
        pickle.dump(DataFrameDict_alt_2, fin)

    if bool(DataFrameDict_alt_3):
        with open('BTB_' + strain + '_id_' + idcko + '_' + stlpec + '_alt_3', "wb") as fin:
            pickle.dump(DataFrameDict_alt_3, fin)


def directory_crawl(root):
    """
    Automaticly crawls through the folders and finds all the files inside.
    passes the file_paths into the load_rat_save function

    Parameters
    ----------
    root : path
        The absolute path to parent folder. The crawler will search all the subfolders.

    Returns
    -------
    None.

    """
    for path, dirnames, filenames in os.walk(root):
        # print(path, dirnames, filenames)
        print(len(filenames))
        if len(filenames) > 0:
            file_names = [os.path.join(path, name) for name in natsort.natsorted(filenames)]
            load_rat_save(file_names)


if __name__ == "__main__":
    directory_crawl(root='C:/Users/ide23/Documents/Diplo/BtB_new')

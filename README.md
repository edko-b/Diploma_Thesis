# Diploma_Thesis
The code used for Classification of Rat Strains Master's Thesis

In this repository are located the scripts used for my master thesis.

In the thesis we used 3 methods.

The first method using the coarse data is written from start to finish in file method1_complete.py. 

The second method using Beat-to-Beat data is separated into:
  1. method2_prepare_dataset.py that was used to handle the complicated structure of the original data.
  2. method2_analysis.py is where the methods described in the paper are programmed.

The last method using the attractor reconstruction technique is again separated into:
  1. method3_prepare_dataset.py, that was used to handle the complicated structure of the original data.
  2. method3_find_optimal.py, where the optimal values of time delay used for embedding is calculated.
  3. method3_analysis.py, where the atrractors are reconstructed, features are transformed and SVM is employed.


Due to the privacy of the data used, THE USED DATA ARE NOT INCLUDED.
method1_complete.py imports data from control_select.py module. In control_select.py, the engine is imported from sql_connection module, that is NOT uploaded to the repo. 
The second and third datasets were stored and loaded from local storage unit, and the datasets are also NOT uploaded to the repo.

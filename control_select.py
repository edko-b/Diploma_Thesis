# -*- coding: utf-8 -*-
"""
Create a pandas dataframe that contains the information about the control phase of the rats.
We only want to use the data of rats when they werent affected by pharmaceutics or altered light/dark phase.
"""
import pandas as pd
from sql_connection import engine

df = pd.read_excel ('Kontrolne_tyzdne.xlsx')
control_list_of_lists = []
for riadok in df.iterrows():
    print(riadok[0])
    one_list = []
    one_list = pd.read_sql(
        "SELECT `idRatExperiment`, `pulsePressure`, `diastolicBP`, `meanBP`," +
        "`systolicBP`, `heartRate`, `activity`, `lightIntensity`" +
        " FROM `parametricData`" +
        " LEFT JOIN ratExperiment on parametricData.idRatExperiment=ratExperiment.id" +
        " WHERE ratExperiment.experimentName like '" + riadok[1][0] +
        "' AND ratExperiment.gender IN (" + riadok[1][1] +
        ") AND DATE(timeStamp) >= '" + riadok[1][2] +
        "' AND DATE(timeStamp) <= '" + riadok[1][3] + "'",
        con=engine,
        parse_dates=[
            'timeStamp'
        ]
    )
    control_list_of_lists.append(one_list)
control_df = pd.concat(control_list_of_lists)



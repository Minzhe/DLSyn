#############################################################################
###                            curate.ALMANIC.py                          ###
#############################################################################

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pkl

proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'

##########################    function   ###########################
def readSingleAsDict(single_path):
    single = pd.read_csv(single_path)
    # print(single['STUDY'].value_counts())
    return single

def calCombExpectation(path, single):
    double = pd.read_csv(path)
    double = double.loc[(double['STUDY'] == 'HNFO00209_15_T72') & (double['PLATE'] == '56_A3_B08'),:]
    double.to_csv(out_path, index=None)
    # print(double)


##############################   main  #################################
double_path = os.path.join(proj_dir, 'data/Curated/NCI.ALMANIC.double.csv')
single_path = os.path.join(proj_dir, 'data/Curated/NCI.ALMANIC.single.csv')
out_path = os.path.join(proj_dir, 'data/Curated/NCI.ALMANIC.test.csv')

single = readSingleAsDict(single_path)
calCombExpectation(double_path, single)
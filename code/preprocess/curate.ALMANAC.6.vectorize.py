#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np
import pickle as pkl

#########################      function     ##########################
def vectorize_array(path, val):
    inpath = path.format(val)
    outpath = inpath.strip('.csv') + '_.pkl'
    data = pd.read_csv(inpath)
    input_array = dummy_input(data[['TYPE', 'CELL', 'COMP1', 'COMP2']])
    output_array = data[['SCORE', 'RATIO_SYN', 'RATIOADJ_SYN', 'SCORE_SYN']]
    with open(outpath, 'wb') as f:
        pkl.dump({'in_array': input_array, 'out_array': output_array}, file=f)


### >>>>>>>>>>>>>>>>>   dummy input   <<<<<<<<<<<<<<<<<<< ###
def dummy_input(df):
    panel = pd.get_dummies(df['TYPE'].apply(lambda x: x.replace(' ', '')))
    cell = pd.get_dummies(df['CELL'])
    comp1 = pd.get_dummies(df['COMP1'].astype(int), prefix='COMP')
    comp2 = pd.get_dummies(df['COMP2'].astype(int), prefix='COMP')
    left1 = list(set(comp2.columns) - set(comp1.columns))
    left2 = list(set(comp1.columns) - set(comp2.columns))
    for col in left1: comp1[col] = 0
    for col in left2: comp2[col] = 0
    comp1 = comp1.sort_index(axis=1)
    comp2 = comp2.sort_index(axis=1)
    comp = comp1 + comp2
    input_array = pd.concat([panel, cell, comp], axis=1)
    return input_array


#########################      main     ##########################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.syn.{}.csv')
vectorize_array(data_path, 'max')
vectorize_array(data_path, 'min')
vectorize_array(data_path, 'mean')
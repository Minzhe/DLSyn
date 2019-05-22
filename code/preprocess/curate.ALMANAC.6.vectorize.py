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
    outpath = (inpath.strip('.csv') + '.pkl').replace('ComboScore', 'train_test_data')
    data = pd.read_csv(inpath)
    cell_array, comp_array = dummy_input(data[['TYPE', 'CELL', 'COMP1', 'COMP2']])
    output_array = data[[col for col in data.columns if col not in ['TYPE', 'CELL', 'COMP1', 'COMP2']]]
    with open(outpath, 'wb') as f:
        pkl.dump({'in_array': input_array, 'out_array': output_array}, file=f)


### >>>>>>>>>>>>>>>>>   dummy input   <<<<<<<<<<<<<<<<<<< ###
def dummy_input(df):
    # cell type
    panel = pd.get_dummies(df['TYPE'].apply(lambda x: x.replace(' ', '')))
    cell = pd.get_dummies(df['CELL'])
    cell_array = pd.concat([panel, cell], axis=0)
    # compound
    comp1 = pd.get_dummies(df['COMP1'], prefix='COMP')
    comp2 = pd.get_dummies(df['COMP2'], prefix='COMP')
    left1 = list(set(comp2.columns) - set(comp1.columns))
    left2 = list(set(comp1.columns) - set(comp2.columns))
    for col in left1: comp1[col] = 0
    for col in left2: comp2[col] = 0
    comp1 = comp1.sort_index(axis=1)
    comp2 = comp2.sort_index(axis=1)
    comp_array = comp1 + comp2
    return cell_array, comp_array


#########################      main     ##########################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.{}.csv')
# vectorize_array(data_path, 'max')
# vectorize_array(data_path, 'min')
# vectorize_array(data_path, 'mean')
vectorize_array(data_path, 'doseagg_min')
# vectorize_array(data_path, 'min_max_mean')
# vectorize_array(data_path, 'dose')
# vectorize_array(data_path, 'mean.norm_True')

####################################################################################
###                              curate.metabolism.py                            ###
####################################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import utility.utility as util

##########################     function    ##########################
def curate_metabolomics(path, cell):
    data = pd.read_csv(path)
    data['CCLE_ID'] = data['CCLE_ID'].apply(lambda x: util.cleanCellName(x.split('_')[0]))
    data = data.loc[data['CCLE_ID'].isin(cell),:].drop(['DepMap_ID'], axis=1).set_index('CCLE_ID')
    data.index.name = None
    # cell_nan = list(set(cell) - set(data.index))
    # data = pd.concat([data, pd.DataFrame(np.nan, index=cell_nan, columns=data.columns)])
    return data.sort_index(axis=0).sort_index(axis=1)

def get_nci60(data_path):
    cell = np.unique(pd.read_csv(data_path, usecols=['CELL'], squeeze=True))
    cell = list(map(util.cleanCellName, cell))
    return cell


##########################      main     ##########################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.doseagg_min.csv')
meta_path = os.path.join(proj_dir, 'data/DepMap/CCLE_metabolomics_20190502.csv')
out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/CCLE_metabolomics.csv')

cell = get_nci60(data_path)
data = curate_metabolomics(meta_path, cell)
data.to_csv(out_path)

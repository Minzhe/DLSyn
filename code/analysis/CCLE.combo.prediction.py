######################################################################################
###                           CCLE.combo.prediction.py                             ###
######################################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
# proj_dir = 'Z:/bioinformatics/s418336/projects/DLSyn'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import pickle as pkl
import utility.utility as util

#######################################      fucntion     #########################################
def summarize_ccle_cell(pred_dir, ccle_path):
    cells = list(map(lambda x: x.split('.')[0], os.listdir(pred_dir)))
    ccle = pd.read_csv(ccle_path, usecols=['CCLE_Name', 'Primary Disease', 'Subtype Disease'])
    ccle['CCLE_Name'] = ccle['CCLE_Name'].apply(lambda x: x.split('_')[0])
    ccle = ccle.loc[ccle['CCLE_Name'].isin(cells),:]
    ccle = ccle.sort_values(by=['Primary Disease', 'Subtype Disease', 'CCLE_Name'])
    return ccle

def check_almanac_consistency(almanac_path, pred_dir):
    # compound
    comp_index = util.read_compound_index()
    # truth
    almanac = pd.read_csv(almanac_path, usecols=['CELL', 'COMP1', 'COMP2', 'RATIOADJ_SYN'])
    cells = np.unique(almanac['CELL'])
    # prediction
    pred = []
    for cell in cells:
        tmp_path = os.path.join(pred_dir, '{}.csv'.format(cell))
        if os.path.isfile(tmp_path):
            pred.append(pd.read_csv(tmp_path)) 
    pred = pd.concat(pred).rename(columns={'SYN': 'PRED'})
    pred['COMP1'] = pred['COMP1'].apply(lambda x: comp_index[x])
    pred['COMP2'] = pred['COMP2'].apply(lambda x: comp_index[x])
    # merge
    pred = pred.merge(almanac)
    return pred



###################################     main    ######################################
pred_dir = os.path.join(proj_dir, 'result/ALMANAC.prediction/CCLE.prediction/Cell')
ccle_path = os.path.join(proj_dir, 'data/DepMap/DepMap.celllines.2019.q1.csv')
almanac_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.doseagg_min.csv')
data_path = data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/train_test_data/curated.combo.syn.cancer_gene.ccle_norm.pkl')

# cell info
# out_path = os.path.join(proj_dir, 'result/ALMANAC.prediction/CCLE.prediction/cell.info.csv')
# cell = summarize_ccle_cell(pred_dir, ccle_path)
# cell.to_csv(out_path, index=None)

# check almanac
out_path = os.path.join(proj_dir, 'result/ALMANAC.prediction/CCLE.prediction/ALMANAC.prediction.csv')
data = check_almanac_consistency(almanac_path, pred_dir)
data.to_csv(out_path, index=None)
####################################################################################
###                              curate.NCI60.expr.py                            ###
####################################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import re
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import utility.utility as util

##############################    function   ################################
def read_expr(data_path, cell_path, expr_path, filter='all_gene'):
    cell = np.unique(pd.read_csv(data_path, usecols=['CELL'], squeeze=True))
    cell = list(map(util.cleanCellName, cell))
    cell_id = pd.read_csv(cell_path, usecols=['DepMap_ID', 'CCLE_Name'])
    cell_id['CCLE_Name'] = cell_id['CCLE_Name'].apply(lambda x: x.split('_')[0].replace('NIH', ''))
    cell_id = cell_id.loc[cell_id['CCLE_Name'].isin(cell),:]
    cell_id = {row['DepMap_ID']:row['CCLE_Name'] for _, row in cell_id.iterrows()}
    # expression
    expr = pd.read_csv(expr_path, index_col=0)
    expr = expr.loc[expr.index.to_series().isin(cell_id.keys()),:]
    expr.index = expr.index.to_series().apply(lambda x: cell_id[x])
    expr.columns = expr.columns.to_series().apply(lambda x: x.split(' ')[0])
    if filter =='cancer_gene':
        genes = util.read_cancer_gene()
        expr = expr.loc[:,expr.columns.to_series().isin(genes)]
    expr = expr.loc[:,(expr > 0).sum() / expr.shape[0] > 0.2]
    expr = expr.sort_index(axis=0).sort_index(axis=1)
    # oupath
    out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/curated.expr.{}.csv'.format(filter))
    expr.to_csv(out_path)

#############################      main     ###############################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.syn.doseagg_min.csv')
cell_path = os.path.join(proj_dir, 'data/DepMap/DepMap.celllines.2019.q1.csv')
expr_path = os.path.join(proj_dir, 'data/DepMap/CCLE_RNAseq_TPM_19Q1.csv')
out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/curated.expr.csv')
read_expr(data_path, cell_path, expr_path, filter='cancer_gene')
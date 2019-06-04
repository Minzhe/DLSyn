######################################################################################
###                              curate.NCI60.crispr.py                            ###
######################################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import re
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import utility.utility as util

##########################     function    ##########################
def curate_genomic(crispr_path, cell):
    crispr = pd.read_csv(crispr_path, index_col=0)
    crispr = crispr.loc[crispr.index.to_series().isin(cell['DepMap_ID']),:]
    crispr.columns = crispr.columns.to_series().apply(lambda x: x.split(' ')[0])
    genes = util.read_cancer_gene()
    crispr = crispr.loc[:,crispr.columns.to_series().isin(genes)]
    crispr.index = crispr.index.to_series().apply(lambda x: cell['CCLE_Name'][cell['DepMap_ID'] == x].values[0])
    return crispr.sort_index().sort_index(axis=1)

def get_nci60(data_path, ccle_path):
    cell = np.unique(pd.read_csv(data_path, usecols=['CELL'], squeeze=True))
    cell = list(map(util.cleanCellName, cell))
    ccle = pd.read_csv(ccle_path, usecols=['DepMap_ID', 'CCLE_Name'])
    ccle['CCLE_Name'] = ccle['CCLE_Name'].apply(lambda x: util.cleanCellName(x.split('_')[0]))
    cell = ccle.loc[ccle['CCLE_Name'].isin(cell),:].sort_values(by=['CCLE_Name'])
    return cell

def get_ccle(ccle_path):
    cell_id = pd.read_csv(ccle_path, usecols=['DepMap_ID', 'CCLE_Name'])
    cell_id['CCLE_Name'] = cell_id['CCLE_Name'].apply(lambda x: x.split('_')[0].replace('NIH', ''))
    cell_id = cell_id.loc[~cell_id['CCLE_Name'].str.contains('MERGE'),:]
    return cell_id


##########################      main     ##########################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.doseagg_min.csv')
ccle_path = os.path.join(proj_dir, 'data/DepMap/DepMap.celllines.2019.q1.csv')
crispr_path = os.path.join(proj_dir, 'data/DepMap/DepMap.CRISPRi.score.csv')
out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/CCLE_crispr.cancer_gene.csv')

# cell = get_nci60(data_path, ccle_path)
cell = get_ccle(ccle_path)
# crispr
# data = curate_genomic(crispr_path, cell)
# data.to_csv(out_path)

# expr
expr_path = os.path.join(proj_dir, 'data/DepMap/CCLE_RNAseq_TPM_19Q1.csv')
out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/CCLE_expr.cancer_gene.csv')
data = curate_genomic(expr_path, cell)
data.to_csv(out_path)

# cnv
# cnv_path = os.path.join(proj_dir, 'data/DepMap/DepMap.CNV.19.q1.csv')
# out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/CCLE_cnv.cancer_gene.csv')
# data = curate_genomic(cnv_path, cell)
# data.to_csv(out_path)
######################################################################################
###                              curate.NCI60.GDSC.py                              ###
######################################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import re
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import utility.utility as util

###############################      function     ##############################
def curated_GDSC(gdsc_path, cell):
    gdsc = pd.read_csv(gdsc_path, usecols=['CELL_LINE_NAME', 'DRUG_ID', 'DRUG_NAME', 'LN_IC50'])
    gdsc['CELL_LINE_NAME'] = gdsc['CELL_LINE_NAME'].apply(lambda x: util.cleanCellName(x))
    gdsc = gdsc.loc[gdsc['CELL_LINE_NAME'].isin(cell['CCLE_Name']),:]
    gdsc['DRUG_NAME'] = list(map(lambda x: x[0] + '.' + str(x[1]), zip(gdsc['DRUG_NAME'], gdsc['DRUG_ID'])))
    gdsc = gdsc.drop(['DRUG_ID'], axis=1)
    gdsc.columns = ['CELL', 'DRUG', 'LOGIC50']
    gdsc = gdsc.pivot(index='CELL', columns='DRUG', values='LOGIC50')
    return gdsc

def get_nci60(data_path, ccle_path):
    cell = np.unique(pd.read_csv(data_path, usecols=['CELL'], squeeze=True))
    cell = list(map(util.cleanCellName, cell))
    ccle = pd.read_csv(ccle_path, usecols=['DepMap_ID', 'CCLE_Name'])
    ccle['CCLE_Name'] = ccle['CCLE_Name'].apply(lambda x: util.cleanCellName(x.split('_')[0]))
    cell = ccle.loc[ccle['CCLE_Name'].isin(cell),:].sort_values(by=['CCLE_Name'])
    return cell

###############################       main      ################################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.doseagg_min.csv')
ccle_path = os.path.join(proj_dir, 'data/DepMap/DepMap.celllines.2019.q1.csv')
gdsc_path = os.path.join(proj_dir, 'data/GDSC/GDSC.fitted_dose_response.v17.3.csv')
out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/GDSC_drugresp.csv')

cell = get_nci60(data_path, ccle_path)
data = curated_GDSC(gdsc_path, cell)
data.to_csv(out_path)
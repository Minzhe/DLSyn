#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np

##############################       function     ###########################
def cal_single_eff(path, out_path):
    single = pd.read_csv(path)
    single['RATIO'] = round(single['TVAL'] / single['CVAL'], 6)
    single['RATIOADJ'] = round((single['TVAL'] - single['T0VAL']) / (single['CVAL'] - single['T0VAL']), 6)
    single = single.groupby(by=['TYPE', 'CELL', 'COMP1', 'CONC1'], as_index=False)[['RATIO', 'RATIOADJ']].median()
    single = single.sort_values(by=['COMP1', 'CELL', 'TYPE', 'CONC1'])
    single.to_csv(out_path, index=None)

def cal_min_single_eff(path):
    single = pd.read_csv(path)
    single = single.groupby(by=['TYPE', 'CELL', 'COMP1'], as_index=False).min().sort_values(by=['COMP1', 'TYPE', 'CELL'])
    single = single.drop(['CONC1'], axis=1)
    out_path = path.strip('csv') + 'doseagg_min.csv'
    single.to_csv(out_path, index=None)

################################     main    #################################
single_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/raw.single.growth.csv')
out_path    = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.single.score.csv')

# cal_single_eff(single_path, out_path)
cal_min_single_eff(out_path)
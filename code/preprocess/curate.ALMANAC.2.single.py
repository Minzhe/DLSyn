#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np

##############################       function     ###########################
def cal_single_eff(path):
    single = pd.read_csv(path)
    single['RATIO'] = round(single['TVAL'] / single['CVAL'], 6)
    single['RATIOADJ'] = round((single['TVAL'] - single['T0VAL']) / (single['CVAL'] - single['T0VAL']), 6)
    single = single.groupby(by=['TYPE', 'CELL', 'COMP1', 'CONC1'], as_index=False)[['RATIO', 'RATIOADJ']].median()
    return single.sort_values(by=['COMP1', 'CELL', 'TYPE', 'CONC1'])


################################     main    #################################
single_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/raw.single.growth.csv')
out_path    = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.single.score.csv')

single = cal_single_eff(single_path)
single.to_csv(out_path, index=None)
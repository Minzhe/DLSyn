#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np

##############################       function     ###########################
def cal_combo_eff(path, out_path):
    combo = pd.read_csv(path)
    combo['RATIO'] = round(combo['TVAL'] / combo['CVAL'], 6)
    combo['RATIOADJ'] = round((combo['TVAL'] - combo['T0VAL']) / (combo['CVAL'] - combo['T0VAL']), 6)
    combo = combo.groupby(by=['TYPE', 'CELL', 'COMP1', 'CONC1', 'COMP2', 'CONC2'], as_index=False, sort=False)[['RATIO', 'RATIOADJ', 'EXPECTED', 'SCORE']].median()
    combo.to_csv(out_path, index=None)

def cal_min_combo_eff(path):
    combo = pd.read_csv(path)
    combo = combo.groupby(by=['TYPE', 'CELL', 'COMP1', 'COMP2'], as_index=False, sort=False).min().sort_values(by=['COMP1', 'COMP2', 'TYPE', 'CELL'])
    combo = combo.drop(['CONC1', 'CONC2'], axis=1)
    out_path = path.strip('csv') + 'doseagg_min.csv'
    combo.to_csv(out_path, index=None)


################################     main    #################################
combo_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/raw.combo.growth.csv')
out_path   = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.score.csv')
# cal_combo_eff(combo_path, out_path)
cal_min_combo_eff(out_path)
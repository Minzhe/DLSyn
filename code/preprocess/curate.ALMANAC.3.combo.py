#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np

##############################       function     ###########################
def cal_combo_eff(path):
    combo = pd.read_csv(path)
    combo['RATIO'] = round(combo['TVAL'] / combo['CVAL'], 6)
    combo['RATIOADJ'] = round((combo['TVAL'] - combo['T0VAL']) / (combo['CVAL'] - combo['T0VAL']), 6)
    combo = combo.groupby(by=['TYPE', 'CELL', 'COMP1', 'CONC1', 'COMP2', 'CONC2'], as_index=False, sort=False)[['RATIO', 'RATIOADJ', 'EXPECTED', 'SCORE']].median()
    return combo


################################     main    #################################
combo_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/raw.combo.growth.csv')
out_path   = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.score.csv')
combo = cal_combo_eff(combo_path)
combo.to_csv(out_path, index=None)
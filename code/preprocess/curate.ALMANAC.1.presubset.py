#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import utility.utility as util

##############################    function   ################################
def read_combo_score(path):
    print('Reading combination growth ...')
    combo = pd.read_csv(path, usecols=['NSC1', 'CONCINDEX1', 'CONC1', 'NSC2', 'CONCINDEX2', 'CONC2', 'TESTVALUE', 'CONTROLVALUE', 'TZVALUE', 'EXPECTEDGROWTH', 'SCORE', 'PANEL', 'CELLNAME'])
    combo = combo.rename(columns={'TESTVALUE': 'TVAL', 'CONTROLVALUE': 'CVAL', 'TZVALUE': 'T0VAL', 'PANEL': 'TYPE', 'CELLNAME': 'CELL', 'NSC1': 'COMP1', 'NSC2': 'COMP2', 'EXPECTEDGROWTH': 'EXPECTED'})
    combo['CONC1'] = 1e6 * combo['CONC1']
    combo['CONC2'] = 1e6 * combo['CONC2']
    combo['EXPECTED'] = combo['EXPECTED'] / 100
    combo['SCORE'] = combo['SCORE'] / 100
    combo = combo[['TYPE', 'CELL', 'COMP1', 'CONCINDEX1', 'CONC1', 'COMP2', 'CONCINDEX2', 'CONC2', 'TVAL', 'CVAL', 'T0VAL', 'EXPECTED', 'SCORE']]
    # get single and combo
    single, combo = get_single_combo(combo)
    # sort compound
    combo = sort_comp(combo)
    return single, combo

# def read_combo_score(path):
#     print('Reading combination growth ...')
#     combo = pd.read_csv(path, sep='\t', usecols=['NSC1', 'CONCINDEX1', 'CONC1', 'NSC2', 'CONCINDEX2', 'CONC2', 'TESTVALUE', 'CONTROLVALUE', 'TZVALUE', 'EXPECTEDGROWTH', 'SCORE', 'PANEL', 'CELLNAME'])
#     combo = combo.rename(columns={'TESTVALUE': 'TVAL', 'CONTROLVALUE': 'CVAL', 'TZVALUE': 'T0VAL', 'PANEL': 'TYPE', 'CELLNAME': 'CELL', 'NSC1': 'COMP1', 'NSC2': 'COMP2', 'EXPECTEDGROWTH': 'EXPECTED'})
#     combo = combo.loc[~(combo['TVAL'] == 'TESTVALUE'),:]
#     # convert type
#     print('Converting types ...')
#     combo['CELL'] = combo['CELL'].apply(util.cleanCellName)
#     combo['COMP1'] = combo['COMP1'].astype(float)
#     combo['COMP2'] = combo['COMP2'].astype(float)
#     combo['CONC1'] = 1e6 * combo['CONC1'].astype(float)
#     combo['CONC2'] = 1e6 * combo['CONC2'].astype(float)
#     combo['CONCINDEX1'] = combo['CONCINDEX1'].astype(float)
#     combo['CONCINDEX2'] = combo['CONCINDEX2'].astype(float)
#     combo['TVAL'] = combo['TVAL'].astype(float)
#     combo['CVAL'] = combo['CVAL'].astype(float)
#     combo['T0VAL'] = combo['T0VAL'].astype(float)
#     combo['EXPECTED'] = combo['EXPECTED'].astype(float) / 100
#     combo['SCORE'] = combo['SCORE'].astype(float) / 100
#     combo = combo.loc[(combo['COMP1'] == 750) & (combo['COMP2'] == 740), ]
#     combo.to_csv('test.csv', index=None)
#     exit()
#     combo = combo[['TYPE', 'CELL', 'COMP1', 'CONCINDEX1', 'CONC1', 'COMP2', 'CONCINDEX2', 'CONC2', 'TVAL', 'CVAL', 'T0VAL', 'EXPECTED', 'SCORE']]
#     single, combo = get_single_combo(combo)
#     # sort compound
#     combo = sort_comp(combo)
#     return single, combo

### >>>>>>>>>>>>>>>>>>>>>> utility <<<<<<<<<<<<<<<<<<<<<< ###
def sort_comp(combo):
    print('Sorting compound order ...')
    combo = combo.groupby(by=['COMP1', 'COMP2'], as_index=False).apply(sort_pair)
    combo = combo.sort_values(by=['COMP1', 'COMP2', 'TYPE', 'CELL', 'CONC1', 'CONC2'])
    combo.index = range(combo.shape[0])
    return combo

def sort_pair(df):
    if np.unique(df['COMP1'])[0] > np.unique(df['COMP2'])[0]:
        df = df.rename(columns={'COMP1': 'COMP2', 'COMP2': 'COMP1', 'CONCINDEX1': 'CONCINDEX2', 'CONCINDEX2': 'CONCINDEX1', 'CONC1': 'CONC2', 'CONC2': 'CONC1'})
    return df[['TYPE', 'CELL', 'COMP1', 'CONCINDEX1', 'CONC1', 'COMP2', 'CONCINDEX2', 'CONC2', 'TVAL', 'CVAL', 'T0VAL', 'EXPECTED', 'SCORE']]

def get_single_combo(combo):
    print('Subseting single and combo treatment ...')
    single = combo.loc[combo['COMP2'].isnull(),['TYPE', 'CELL', 'COMP1', 'CONCINDEX1', 'CONC1', 'TVAL', 'CVAL', 'T0VAL']].sort_values(by=['COMP1', 'CELL', 'CONC1'])
    double = combo.loc[combo['COMP2'].notnull(),:]
    return single, double


##############################    main   ################################
data_path = os.path.join(proj_dir, 'data/NCI.ALMANAC/ComboDrugGrowth_Nov2017.csv')
combo_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/raw.combo.growth.csv')
single_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/raw.single.growth.csv')

single, combo = read_combo_score(data_path)
single.to_csv(single_path, index=None)
combo.to_csv(combo_path, index=None)
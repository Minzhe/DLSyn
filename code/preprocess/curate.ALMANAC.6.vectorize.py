#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import pickle as pkl
import warnings
from sklearn.preprocessing import StandardScaler
import utility.utility as util

#########################      function     ##########################
def vectorize_array(path, filter):
	# outpath = (path.strip('.csv') + '.{}.pkl'.format(filter)).replace('ComboScore', 'train_test_data')
	outpath = (path.strip('.csv') + '.{}.ccle_norm.pkl'.format(filter)).replace('ComboScore', 'train_test_data')
	# mut_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/DNA_Exome_Seq_protein_function_affecting.{}.csv'.format(filter))
	mut_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/CCLE_mutation.min_1.{}.csv'.format(filter))
	# expr_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/RNA_seq_composite_expression.{}.csv'.format(filter))
	expr_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/CCLE_expr.{}.csv'.format(filter))
	protein_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/SWATH_Protein.{}.csv'.format(filter))
	meta_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/CCLE_metabolomics.{}.csv'.format(filter))
	crispr_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/CCLE_crispr.{}.csv'.format(filter))
	gdsc_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/GDSC_drugresp.csv')
	data = pd.read_csv(path)
	data['COMP1'] = data['COMP1'].astype(int)
	data['COMP2'] = data['COMP2'].astype(int)
	# cell line, compound encoding
	panel_array, panel_index, cell_array, cell_index, comp_array, comp_index = dummy_input(data[['TYPE', 'CELL', 'COMP1', 'COMP2']])
	# genomic encoding
	expr_array, expr_index, expr_scaler          = read_genomic(data['CELL'], expr_path, 'expr')
	mut_array, mut_index, _                      = read_genomic(data['CELL'], mut_path, 'mut')
	protein_array, protein_index, protein_scaler = read_genomic(data['CELL'], protein_path, 'protein')
	meta_array, meta_index, meta_scaler          = read_genomic(data['CELL'], meta_path, 'meta')
	crispr_array, crispr_index, crispr_scaler    = read_genomic(data['CELL'], crispr_path, 'crispr')
	ic50_array, ic50_index, ic50_scaler          = read_genomic(data['CELL'], gdsc_path, 'ic50')
	# output
	output_array = data[[col for col in data.columns if col not in ['TYPE', 'CELL', 'COMP1', 'COMP2']]]
	with open(outpath, 'wb') as f:
		pkl.dump({'panel_array': panel_array,     'panel_index': panel_index,
				  'cell_array': cell_array,       'cell_index': cell_index,
				  'comp_array': comp_array,       'comp_index': comp_index,
				  'expr_array': expr_array,       'expr_index': expr_index,       'expr_scaler': expr_scaler,
				  'mut_array': mut_array,         'mut_index': mut_index,
				  'protein_array': protein_array, 'protein_index': protein_index, 'protein_scaler': protein_scaler,
				  'meta_array': meta_array,       'meta_index': meta_index,       'meta_scaler': meta_scaler,
				  'crispr_array': crispr_array,   'crispr_index': crispr_index,   'crispr_scaler': crispr_scaler,
				  'ic50_array': ic50_array,       'ic50_index': ic50_index,       'ic50_scaler': ic50_scaler,
				  'output_array': output_array}, file=f)


### >>>>>>>>>>>>>>>>>   dummy input   <<<<<<<<<<<<<<<<<<< ###
def dummy_input(df):
	# cell type
	panel = pd.get_dummies(df['TYPE'])
	panel_index = util.index_array(panel.columns)
	cell = pd.get_dummies(df['CELL'])
	cell_index = util.index_array(cell.columns)
	# compound
	comp1 = pd.get_dummies(df['COMP1'], prefix='COMP')
	comp2 = pd.get_dummies(df['COMP2'], prefix='COMP')
	left1 = list(set(comp2.columns) - set(comp1.columns))
	left2 = list(set(comp1.columns) - set(comp2.columns))
	for col in left1: comp1[col] = 0
	for col in left2: comp2[col] = 0
	comp1 = comp1.sort_index(axis=1)
	comp2 = comp2.sort_index(axis=1)
	comp = comp1 + comp2
	comp_index = util.index_array(comp.columns)
	return np.array(panel), panel_index, np.array(cell), cell_index, np.array(comp), comp_index

def read_genomic(cell, path, data_type, impute_cell=False):
	print('Reading {} ...'.format(path))
	data = pd.read_csv(path, index_col=0)
	# unfound cell
	unfound = np.unique(cell)[np.where(list(map(lambda x: x not in list(data.index), np.unique(cell))))[0]]
	if len(unfound) > 0:
		print('Cell {} {} information not found.'.format(unfound, data_type))
		if impute_cell:
			tmp = data.T
			for col in unfound:
				tmp[col] = np.nan
			data = tmp.T
	# normalize data
	if data_type in ['expr', 'protein', 'meta', 'crispr', 'ic50']:
		scaler = StandardScaler().fit(data)
		data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
	elif data_type == 'mut':
		data = data.fillna(0)
		scaler = None
	# get entry
	data, index = fill_table(cell, data)
	return data, index, scaler

def fill_table(arr, df):
	mat = np.full(fill_value=np.nan, shape=(len(arr), len(df.columns)))
	for x in np.unique(arr):
		if x in df.index:
			mat[np.where(arr == x)[0],:] = df.T[x]
	index = util.index_array(df.columns)
	return mat, index



#########################      main     ##########################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.doseagg_min.csv')
vectorize_array(data_path, filter='cancer_gene')
#############################################################################
###                           predict.DrugSyn.py                          ###
#############################################################################

proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import r2_score
from model import neural_net
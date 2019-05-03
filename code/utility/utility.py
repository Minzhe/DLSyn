#############################################################################
###                               utility.py                              ###
#############################################################################

import re

### >>>>>>>>>>>>>>>>>>>>>>>>>>>>   clean string  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ###
def cleanCellName(name):
    '''
    Get alias of cell line name.
    '''
    alias = re.sub(r'\(.*\)', '', name)
    alias = re.sub(r'[-_\[\]]', '', alias)
    alias = re.sub(r' ', '', alias)
    return alias.upper()
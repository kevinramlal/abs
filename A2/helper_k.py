#Kevin's Helper functions for A2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 

sys.path.insert(1,'../A1')
from fixed_income import *  #retreives FixedIncome class from A1 folder 


dynamic = pd.read_csv('./Given_Files/dynamic.csv',  low_memory = False)
static = pd.read_csv('./Given_Files/static.csv',low_memory = False)


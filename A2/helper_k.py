#Kevin's Helper functions for A2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 

sys.path.insert(1,'../A1')
from fixed_income import *  #retreives FixedIncome class from A1 folder 
from homework1 import * #imports homework1 class where we simulated interest rate paths
from hazard_kevin_copy import *

hw1 = Homework1(show_prints = False,show_plots = False)
hw1.fit_term_structure_model()
simulated_rates, simulated_Z = hw1.simulate_interest_rates(n=100)


dynamic = pd.read_csv('./Given_Files/dynamic.csv',  thousands = ",",low_memory = False)
static = pd.read_csv('./Given_Files/static.csv', thousands = ",", low_memory = False)

hz = Hazard(static, prepay_col="prepay", end_col="period_end",beg_col="period_begin", end_max=60, cov_cols=["cpn_gap", "summer"])
#hz.fit_parameters_brute()
hz.fit_parameself_t_allrs_grad()
#hz.baseline_hazard()
hz.parameters_hessian()
hz.parameters_se()

print(dynamic.head())
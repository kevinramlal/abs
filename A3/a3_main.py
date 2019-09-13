
# Public libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import os 
import sys 

sys.path.insert(1,'./Given_Files')
# Private libraries
import homework1
import hazard
import remic
import a3_helper

"""
Step 1: Need to estimate 4 hazards {ARM/FRM} x {Prepayment/Default}
Prepayment is the same as the coupon gap in HW2
Default is defined as LTV : (remaining balance)/(home price)
"""

arm_data = pd.read_csv('./Given_Files/ARM_perf.csv')
frm_data = pd.read_csv('./Given_Files/FRM_perf.csv')

#To use our class function for calculating hazards - we need to get our data to be similiar to that of HW2
#This innvolves consolidating data per episode

hlp = a3_helper.h3_helper()
test = hlp.data_wrangler(arm_data)
print(test.head()) #NEED TO GET COUPON GAP? AND DTI?

# arm_prepay = hazard.Hazard(hz_static_data, prepay_col="Prepayment_indicator", end_col="period_end",\
# beg_col="", end_max=60, cov_cols=["Spread", "spring_summer"], tables_file=tables_file, show_prints=True, show_plo
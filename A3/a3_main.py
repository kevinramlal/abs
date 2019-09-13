
# Public libraries
import pandas as pd
import os 
import sys 

sys.path.insert(1,'./Given_Files')
# Private libraries
import homework1
import hazard
import remic

"""
Step 1: Need to estimate 4 hazards {ARM/FRM} x {Prepayment/Default}
Prepayment is the same as the coupon gap in HW2
Default is defined as LTV : (remaining balance)/(home price)
"""

arm_data = pd.read_csv('./Given_Files/ARM_perf')
frm_data = pd.read_csv('./Given_Files/FRM_perf')
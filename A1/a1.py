"""///
ABS Assignment 1
Members: Kevin, Nico, Romain, Sherry, Trilok
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import datetime
#Helper Functions
def get_days_act_360(start_date,end_date):
    return (end_date-start_date).days/360
    
def get_days_30I_360(start_date,end_date):
    Y1 = start_date.year
    Y2 = end_date.year
    M1 = start_date.month
    M2 = end_date.month
    D1 = start_date.day
    D2 = end_date.day
    if D1==31:
        D1=30
    if D2==31:
        D2=30
    return (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1))/360

#----Part 1: Importing Data and Calibrating Data-----#
fwds = pd.read_csv('fwds_20040830.csv',header = None, names = ['Fwds'])
zero = pd.read_csv('zero_rates_20040830.csv',header = None, names = ['Zero'])
zr = np.array(zero['Zero'])
zr_temp = np.append([None],zr) #Need to have initial value of None since dates start at 9_01_2004
master_rates = pd.DataFrame()
master_rates['Zero'] = zr_temp

atm_cap = pd.read_csv('atm_cap.csv', header = None) #extracted from 20040830_usd23_atm_caps file
atm_cap.columns = ['Maturity', 'Black Imp Vol']

#Import date column and format 
dates = pd.read_csv('dates.csv',header = None, dtype = str) #from Bloomberg files USD23 tab
dates = pd.to_datetime(dates[0])
dates.columns = ['Date']

#-----------------------------------------------------------------
#a) Caclulate Discount factors Z(D_i)
#-----------------------------------------------------------------

#Combine dates and zero rates 
start_date = dates[0]
master_rates['Dates'] =np.array(dates) #We don't need 2004-09-01 for zeros
master_rates['T_30_360'] = np.array(dates.apply(lambda x: get_days_30I_360(start_date,x)))
master_rates['Discount'] = 1/(1+(master_rates['Zero']/100)/2)**(2*master_rates['T_30_360'])

print(master_rates)
print("a) Discount Factors \n", master_rates[['Dates','Zero','Discount']].head(25), "\n")

#-----------------------------------------------------------------
#b) Calculate quarterly-compounded forward rates between each maturity
#-----------------------------------------------------------------

#Get ACT/360 Convention
master_rates['T_ACT_360'] = np.array(dates.apply(lambda x: get_days_act_360(start_date,x))) 

#T_i - T_i-1 where T_i is ACT/360 convention
master_rates['Tau'] = master_rates['T_ACT_360'].diff(1)

#Forward Rates 
forwards = np.array((1/master_rates['Tau'])*\
						((-master_rates['Discount'].diff(1))/master_rates['Discount']))
master_rates['Forward'] = forwards

print("b) Forward Rates \n", master_rates[['Dates','Discount','Forward']].head(25), "\n")
#-----------------------------------------------------------------
#c) Calculating the at-the-money (ATM) strike rates for each of the 15 caps
#-----------------------------------------------------------------

cap_master_df = pd.DataFrame()
cap_master_df['Maturity'] = atm_cap['Maturity']

strike = []

#The following code corresponds to the equation for X_n in c)
for m in cap_master_df['Maturity']:
	x_n = np.sum(master_rates['Tau'][2:4*m +1]*\
			master_rates['Forward'][2:4*m +1]*master_rates['Discount'][2:4*m +1])/\
			np.sum(master_rates['Tau'][2:4*m +1]*master_rates['Discount'][2:4*m +1])
	strike.append(x_n)


cap_master_df['ATM Strike'] = strike

print("c) ATM Strike Rates vs Maturity \n", cap_master_df, "\n")


#-----------------------------------------------------------------
#d) Estimating k and sigma
#-----------------------------------------------------------------

#Notes
#implement Black's Formula 
#For this question we access forward libor rates and flat vols
#Caps have quarterly caplets in this question
#summing over tau's from 0 to expiry (multiply Maturity by 4)
#use the same flat vol for all quarterly caplets when calculating a specific cap


flat_vols = np.array(atm_cap['Black Imp Vol'])
print(flat_vols)

forward_libor = np.array(fwds['Fwds'])
print(forward_libor,len(forward_libor)) #should be 30

cap_master_df['Flat_Vol'] = flat_vols

def d_1_2(flat_vol,forward_libor,t,strike,type = 1):
	d1 = (1/flat_vol*np.sqrt(t))*np.log(forward/strike) + 0.5*flat_vol*np.sqrt(t)
	if type == 1:
		return d1
	elif type == 2:
		return d1 - flat_vol*np.sqrt(t)

# def caplet(master_rates,t_i,)


if __name__ == '__main__':
		print(master_rates[['T_30_360','T_ACT_360']])

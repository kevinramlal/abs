"""///
ABS Assignment 1
Members: Kevin, Nico, Romain, Sherry, Trilok

#THIS FILE WAS UPLOADED USING GIT
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import datetime
from scipy.stats import norm
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

#Import Expiry, Settlement Date column and format as datetime
dates_all = pd.read_csv('dates_expiry_settle.csv',dtype = str) #from Bloomberg files USD23 tab
dates = pd.to_datetime(dates_all['Settlement Date']) #settlement dates
dates.columns = ['Date']

dates_settle = pd.to_datetime(dates_all['Expiry Date'])
dates_settle.columns = ['Settle_Date']


#-----------------------------------------------------------------
#a) Caclulate Discount factors Z(D_i)
#-----------------------------------------------------------------

#Combine dates and zero rates 
start_date = dates[0]
master_rates['Dates'] =np.array(dates) #We don't need 2004-09-01 for zeros
master_rates['T_30_360'] = np.array(dates.apply(lambda x: get_days_30I_360(start_date,x)))
master_rates['Discount'] = 1/(1+(master_rates['Zero']/100)/2)**(2*master_rates['T_30_360'])
master_rates['Expiry Dates'] = np.array(dates_settle)
master_rates['Expiry_day_count'] = np.array(dates_settle.apply(lambda x: (x - master_rates['Dates'][0]).days))

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



#---------Implementing Blacks Formula
flat_vols = np.array(atm_cap['Black Imp Vol'])
print(flat_vols)

forward_libor = np.array(fwds['Fwds'])
print(forward_libor,len(forward_libor))
master_rates['Forward_Libor'] = forward_libor #Has one extra entry at end that we cannot compute? 

cap_master_df['Flat_Vol'] = flat_vols

def d_1_2(flat_vol,forward_libor,t,strike,type = 1):
	d1 = (1/(flat_vol*np.sqrt(t)))*np.log(forward_libor/strike) + 0.5*flat_vol*np.sqrt(t)
	if type == 1:
		return d1
	elif type == 2:
		return d1 - flat_vol*np.sqrt(t)

#index by np.arange(1,maturity,4)

def caplet(master_rates,time_index,N, flat_vol, strike):
    """
    master_rates - DataFrame that contains dates, discount rates, and forward rates 
    from above.
    time_index - index based on T_30_360 convenion (example - 0.75 -> 3)
    """
    t_i = master_rates['Expiry_day_count'][time_index]/365
    tau = master_rates['Tau'][time_index +1] 
    fwd = master_rates['Forward'][time_index+1]
    fv = flat_vol/100
    discount = master_rates['Discount'][time_index+1]
    d1 = d_1_2(fv,fwd,t_i,strike)
    d2 = d_1_2(fv,fwd,t_i,strike,type =2)
    phi_d1 = norm.cdf(d1)
    phi_d2 = norm.cdf(d2)
    # print(phi_d1,phi_d1,discount,tau,N)
    # print((N*tau)*discount*(fwd*phi_d1 - strike*phi_d2))
    pv = ((N*tau)*discount*(fwd*phi_d1 - strike*phi_d2))

    return pv

###TEST CASE: MATCHES WITH EXCEL SHEET###
print("-------Test Case: 1 Year Cap--------\n")
maturity = cap_master_df['Maturity'][0]
flat_vol = cap_master_df['Flat_Vol'][0]
strike = cap_master_df['ATM Strike'][0]
caplet_range = np.arange(1,maturity*4)
# print(caplet_range)
caplet_pv = []
for index in caplet_range:
    caplet_pv.append(caplet(master_rates,index,10000000,flat_vol,strike))
print("------Prices of Caplets-------\n",caplet_pv,'\n')
print("1 Year Cap Value : ", round(sum(caplet_pv),2))


if __name__ == '__main__':
    #testing caplet
    maturity = cap_master_df['Maturity'][0]
    flat_vol = cap_master_df['Flat_Vol'][0]
    strike = cap_master_df['ATM Strike'][0]
    caplet_range = np.arange(1,maturity*4)
    caplet_pv = []
    for index in caplet_range:
        caplet_pv.append(caplet(master_rates,index,10000000,flat_vol,strike))
    # print(caplet_pv)
"""///
ABS Assignment 1
Members: Kevin, Nico, Romain, Sherry, Sagnik

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm
from scipy.optimize import minimize
from scipy import optimize
from scipy import interpolate
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Custom made classes
import fixed_income
import remic


def latex_table(df, caption="", label="", index=False):
    return "\\begin{table}[H]\n\centering\n"+df.to_latex(index=index)+"\caption{"+caption+"}\n\label{tab:"+label+"}\n\end{table}"


fi = fixed_income.FixedIncome()

#----Part 1: Importing Data and Calibrating Data-----#
fwds = pd.read_csv('fwds_20040830.csv',header = None, names = ['Fwds'])
# zero = pd.read_csv('zero_rates_20040830.csv',header = None, names = ['Zero'])
##Created Custom Zero rates file that adds in extra zero rate for 2034-12-01
zero = pd.read_csv('zero_rates_custom.csv',header = None, names = ['Zero'])

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
master_rates['T_30_360'] = np.array(dates.apply(lambda x: fi.get_days_30I_360(start_date,x)))
master_rates['Discount'] = 1/(1+(master_rates['Zero']/100)/2)**(2*master_rates['T_30_360'])
master_rates['Expiry Dates'] = np.array(dates_settle)
master_rates['Expiry_day_count'] = np.array(dates_settle.apply(lambda x: (x - master_rates['Dates'][0]).days))

print("\na) Discount Factors \n", master_rates[['Dates','Zero','Discount']].head(25), "\n")
#print(latex_table(master_rates[['Dates','Zero','Discount']], caption="Discount Factors", label="p1a_discount", index=False))

#plt.plot(master_rates['Dates'], master_rates['Discount'], 'b', ms = 6)
#plt.xlabel('Date')
#plt.ylabel('Discount Rate')
#plt.title('Discount (Quarterly)')
#plt.savefig('1a_discount.eps', format='eps')
#plt.show()

#-----------------------------------------------------------------
#b) Calculate quarterly-compounded forward rates between each maturity
#-----------------------------------------------------------------

#Get ACT/360 Convention
master_rates['T_ACT_360'] = np.array(dates.apply(lambda x: fi.get_days_act_360(start_date,x))) 

#T_i - T_i-1 where T_i is ACT/360 convention
master_rates['Tau'] = master_rates['T_ACT_360'].diff(1)

#Forward Rates 
forwards = np.array((1/master_rates['Tau'])*\
                        ((-master_rates['Discount'].diff(1))/master_rates['Discount']))

master_rates['Forward'] = forwards

print("\nb) Forward Rates \n", master_rates[['Dates','Discount','Forward']].head(25), "\n")
##print(latex_table(master_rates[['Dates','Zero','Discount','Forward']], caption="Discount factors and forward rates", label="p1ab", index=False))

#plt.plot(master_rates['Dates'], master_rates['Forward'], 'b', ms = 6, label = "Forward Rate")
#plt.xlabel('Date')
#plt.ylabel('Forward Rate')
#plt.title('Forward Rate')
#plt.savefig('1b_forward.eps', format='eps')
#plt.show()

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

print("\nc) ATM Strike Rates vs Maturity \n", cap_master_df, "\n")
#print(latex_table(cap_master_df, caption="ATM strikes for given caps", label="p1c", index=False))

#-----------------------------------------------------------------
#d) Estimating k and sigma
#-----------------------------------------------------------------

#---------Implementing Blacks Formula---------------------#
flat_vols = np.array(atm_cap['Black Imp Vol'])
forward_libor = np.array(fwds['Fwds'])
fwd_libor_temp = [None]
forward_libor = np.append(fwd_libor_temp,forward_libor)
master_rates['Forward_Libor'] = forward_libor #Has one extra entry at end that we cannot compute? 
cap_master_df['Flat_Vol'] = flat_vols

def d_1_2(flat_vol,forward_libor,t,strike,type = 1):
    d1 = (1/(flat_vol*np.sqrt(t)))*np.log(forward_libor/strike) + 0.5*flat_vol*np.sqrt(t)
    if type == 1:
        return d1
    elif type == 2:
        return d1 - flat_vol*np.sqrt(t)

#index by np.arange(1,maturity,4)

def caplet_black(master_rates,time_index,N, flat_vol, strike):
    """
    master_rates - DataFrame that contains dates, discount rates, and forward rates
    from above.
    time_index - index based on T_30_360 convenion (example - 0.75 -> 3)
    Returns caplet value for a given time based on Black formula.
    """
    t_i = master_rates['Expiry_day_count'][time_index]/365
    tau = master_rates['Tau'][time_index +1]
    fwd = master_rates['Forward_Libor'][time_index+1]/100 #Note used Libor Forward
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

def caplet_hull_white(master_rates,index,N,vol,kappa, strike_input):
    """
    Similiar parameters as Black model but needs constant vol and kappa
    """
    tau = master_rates ['Tau'][index+1]
    t_i = master_rates['Expiry_day_count'][index]/365
    discount = master_rates['Discount'][index+1]
    disc_shift = master_rates['Discount'][index]
    b = (1/kappa)*(1 - np.exp(-kappa*tau))
    sigma_p = b*vol*np.sqrt((1- np.exp(-2*kappa*t_i))/(2*kappa))
    d1 = (1/sigma_p)*np.log(discount/(disc_shift*strike_input)) + sigma_p/2
    d2 = d1 - sigma_p

    V = (N/strike_input)*(strike_input*disc_shift*norm.cdf(-d2) - discount*norm.cdf(-d1))
    return V

def cap_black(master_rates, maturity_index, N, cap_master, vol=False):
    """Returns cap value based on Black formula."""
    maturity = cap_master['Maturity'][maturity_index]
    if vol == False:
        flat_vol = cap_master['Flat_Vol'][maturity_index]
    else:
        flat_vol = vol
    strike = cap_master['ATM Strike'][maturity_index]
    caplet_range = np.arange(1,maturity*4)
    cap_pv = 0
    for index in caplet_range:
        cap_pv += caplet_black(master_rates,index,N,flat_vol,strike)
    return cap_pv

def cap_HW(master_rates,maturity_index, vol, N, cap_master, kappa):
    """Returns cap value based on HW model."""
    maturity = cap_master['Maturity'][maturity_index]
    # flat_vol = cap_master['Flat_Vol'][maturity_index]
    strike = cap_master['ATM Strike'][maturity_index]
    caplet_range = np.arange(1,maturity*4)
    cap_pv = 0
    for index in caplet_range:
        tau = master_rates['Tau'][index+1]
        strike_input = (1/(1+strike*tau))
        clet = caplet_hull_white(master_rates,index,10000000,vol,kappa,strike_input)
        cap_pv += clet

    return cap_pv

def to_minimize(params):
    kappa = params[0]
    fv = params[1]
    res = 0
    for i in range(15):
        err = cap_black(master_rates,i, 10000000,  cap_master_df) - cap_HW(master_rates,i, fv,10000000, cap_master_df, kappa)
        res += err**2
    res = np.sqrt(res)
    return res


##Version A: Run this section instead of Version B see minimizer at work - takes a few mins
#-------------------------------------------------
# optimum = minimize(to_minimize, x0 = [0.20, 0.010])
# print("Optimal Kappa: ", optimum.x[0],"\n", "Optimal Vol: ", optimum.x[1])
# opti_kap = optimum.x[0] 
# opti_vol = optimum.x[1]
#---------------------------------------------------


##Version B: Hardcoded optimized kappa and vol for quicker run-time 
#--------------------------------------------------
#Uncomment this to run code faster (don't need to run optimizer)
opti_kap = 0.11469962
opti_vol = 0.01456547
#------------------------------------------------------


#---------
cap_price_list_black = []
cap_price_list_hull = []
for cap_index in range(len(cap_master_df)):
    maturity = cap_master_df['Maturity'][cap_index]
    flat_vol = cap_master_df['Flat_Vol'][cap_index]
    strike = cap_master_df['ATM Strike'][cap_index]
    caplet_range = np.arange(1,maturity*4)
    caplet_black_pv = []
    caplet_hw_pv = []
    for index in caplet_range:
        caplet_black_pv.append(caplet_black(master_rates,index,10000000,flat_vol,strike))
        
        tau = master_rates['Tau'][index+1]
        strike_input = (1/(1+strike*tau))
        caplet_hw_pv.append(caplet_hull_white(master_rates,index,10000000,opti_vol, opti_kap,strike_input))
    # print('Caplets under Cap Maturity :', maturity,"\n", caplet_pv)
    # print('Price of Cap Maturity: ', maturity, "\n", sum(caplet_pv),"\n")
    cap_price_list_black.append(round(sum(caplet_black_pv),3))
    cap_price_list_hull.append(round(sum(caplet_hw_pv),3))
cap_master_df['Cap Prices - Black'] = cap_price_list_black
cap_master_df['Cap Prices - Hull'] = cap_price_list_hull
print("d) Summary of Cap Pricing", cap_master_df, " \n Optimized Kappa: ", opti_kap, \
    "\n", "Optimized Vol: ", opti_vol)

#--------------------------------------------------------------
#e)  Estimate theta(t) from today t_0 to 30-years with timestep 1/12
#--------------------------------------------------------------

#First we need to interpolate the monthly discount factors from quarterly ones

disc_quart = np.array(master_rates ['Discount']).astype(float)
x_quart = np.array(master_rates['T_ACT_360'])
x_monthly = np.arange(0,30 + (1/12),(1/12))
disc_monthly_f = interpolate.interp1d(x_quart, disc_quart, kind = 'cubic')
disc_monthly = disc_monthly_f(x_monthly)


#plt.plot(x_quart,disc_quart, 'b1', ms = 6, label = 'Quarterly Discount')
#plt.plot(x_monthly,disc_monthly ,'k', label = "Interpolated Monthly Discount")
#plt.title('e) Interpolated Monthly Forward')
#plt.xlabel('Months after 9-01-2004')
#plt.ylabel('Discount')
#plt.legend(loc = 'upper right')
#plt.savefig('1e_discount.eps', format='eps')
#plt.show()

#Calculating Monthly Forwards and derivative using finite differencing
fm = np.array([-(np.log(disc_monthly[i]) - np.log(disc_monthly[i-1]))/(1/12) for i in range(1,len(disc_monthly))]).astype(float)
d_fm = np.array([(fm[i] - fm[i-1])/(1/12) for i in range(1,len(fm))]).astype(float)

kap = opti_kap
vol = opti_vol

#Calculating Theta
theta = d_fm + kap*fm[1:] + ((vol**2)/(2*kap))*(1 - np.exp(-2*kap*(x_monthly[2:])))
# print(theta)
# print(len(theta))

#plt.plot(x_monthly[2:], theta, 'b')
#plt.title('Theta vs. Time')
#plt.xlabel('Years')
#plt.ylabel('Theta')
#plt.savefig('1e_theta.eps', format='eps')
#plt.show()


#--------------------------------------------------------------
#f)  Converting Hull White prices into Black implied volatilities
#--------------------------------------------------------------


HW_fitted_vol = []

def to_solve(vol): #we find the Black volatility such that the Black cap price is equal
    #to the Hull White cap price, and we repeat the process for each cap.
    i = 0 #we manually update it from 0 to 14
    res = []
    res.append(np.abs(cap_black(master_rates, i, 10000000, cap_master_df, vol[i]) - cap_price_list_hull[i]))
    return res

HW_fitted_vol.append(optimize.fsolve(to_solve, 0))
for i in range(len(HW_fitted_vol)):
    HW_fitted_vol[i] = HW_fitted_vol[i][0]
initial_vol = list(cap_master_df['Flat_Vol'])

plt.plot(np.arange(0,15,1),initial_vol, 'b1', ms = 6, label = 'Black 76 Implied volatility')
plt.plot(np.arange(0,15,1),HW_fitted_vol ,'k', label = "1F-HW Calibrated Implied volatility")
plt.title('f) Implied vs actual volatilities')
plt.xlabel('Cap index')
plt.ylabel('Black Implied Volatility (%)')
plt.legend(loc = 'upper right')
plt.savefig('q1F.png')
plt.show()



#-----------------------------------------------------------------------------
#----Part 2: Pricing REMIC bonds-----#

#General info
today = '8/15/2004'
first_payment_date = '9/15/2004'
pool_interest_rate = 0.05

# General information of pools
pools_info = pd.read_csv('pools_general_info.csv', thousands=',')

# General information of classes
classes_info = pd.read_csv('classes_general_info.csv', thousands=',')

# Allocation sequence for principal
principal_sequential_pay = {'1': ['CA','CY'], '2': ['CG','VE','CM','GZ','TC','CZ']}

# Accruals accounts sequence
accruals_sequential_pay = {'GZ': ['VE','CM'], 'CZ': ['CG','VE','CM','GZ','TC']}


# REMIC cash flows
hw_remic = remic.REMIC(today, first_payment_date, pool_interest_rate, pools_info, classes_info, principal_sequential_pay, accruals_sequential_pay)

#float input of PSA
hw_remic.calculate_pool_cf(1.5)


# REMIC pricing
n = 10000
dt = 1/12
kappa = opti_kap
sigma = opti_vol
r0 = fi.hull_white_instantaneous_spot_rate(0, 3*dt, master_rates.loc[1, 'Discount'], theta, kappa, sigma)
simulated_rates = fi.hull_white_simulate_rates_antithetic(n, r0, dt, theta, kappa, sigma)

hw_remic.calculate_classes_cf(simulated_rates)

simulated_Z = fi.hull_white_discount_factors_antithetic_GSI_version(simulated_rates, dt)
P = hw_remic.price_classes(simulated_Z)

#Calculate duration and convexity
delta_r = r0/100
simulated_rates_minus = fi.hull_white_simulate_rates_antithetic(n,r0-delta_r,delta_r, dt, theta, kappa, sigma)
hw_remic.calculate_classes_cf(simulated_rates_minus)
simulated_Z_minus = fi.hull_white_discount_factors_antithetic_GSI_version(simulated_rates_minus, dt)
P_minus = hw_remic.price_classes(simulated_Z_minus)

simulated_rates_plus = fi.hull_white_simulate_rates_antithetic(n,r0+delta_r,delta_r, dt, theta, kappa, sigma)
hw_remic.calculate_classes_cf(simulated_rates_plus)
simulated_Z_plus = fi.hull_white_discount_factors_antithetic_GSI_version(simulated_rates_plus, dt)
P_plus = hw_remic.price_classes(simulated_Z_plus)


duration = fi.calculate_eff_dura_or_convexity(self,r0,P,P_plus,P_minus,if_duration=1)
convexity = fi.calculate_eff_dura_or_convexity(self,r0,P,P_plus,P_minus,if_duration=0)

print('The duration of the bonds are', duration)
print('The convexity of the bonds are', convexity)

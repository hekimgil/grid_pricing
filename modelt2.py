#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model t2 uses the the state data as parameterized by t.
Model parameters are changed to allow a solution within a possible range.
@author: Hakan Hekimgil, Jafar Chaab
"""

import numpy as np

# this is a dummy function, not used...
def retailprice(t):
    return None


# MODEL PARAMETERS
# number of service providers
nproviders = 1
# number of customers
ncustomers = 3
# number of time slots
ntimeslots = 24
# price bounds (p.225 last paragraph)
k1 = 1.5
k2 = 1.5
# weighting factor (p.225 last paragraph)
rho = 0.9
# wholesale price from grid operator
wholepricedata =[
        1.90, 1.80, 1.70, 1.60, 1.60, 1.90,
        2.00, 2.20, 2.30, 2.60, 2.90, 3.30,
        3.40, 3.60, 3.50, 5.40, 5.50, 5.00,
        4.00, 3.20, 3.10, 3.60, 3.00, 2.60]
def wholeprice(t):
    return wholepricedata[t-1]

# CUSTOMER PARAMETERS
# customers' dissatisfaction related parameters (Table 2)
alphan = [0.8, 0.5, 0.3]
betan = [0.1, 0.1, 0.1]
# customer demand data (currentle from Fig. 5)
edemandcritdata =[
        [14.21839963, 13.93480264, 14.10866226, 13.76846875, 
         13.53186914, 12.32905852, 12.29181385, 13.02856704, 
         14.63546306, 16.67846721, 17.74658271, 18.0714176, 
         18.21125787, 18.56504377, 18.62916603, 18.62793734, 
         18.39202887, 17.95016127, 17.48994010, 17.79181385, 
         17.07533405, 15.99324221, 15.14636769, 14.64966979], 
        [14.20849332, 13.93142374, 13.77929658, 14.15780986, 
         14.78474889, 15.45492244, 18.60720319, 22.92251574, 
         26.89663646, 29.62885885, 31.39179849, 31.80333282, 
         32.41890647, 33.31884503, 33.17961911, 31.38050991, 
         27.65535248, 23.47527262, 21.13914913, 20.69551528, 
         18.78382737, 16.88465673, 15.64168331, 15.09629857], 
        [14.80333282, 14.55052987, 14.44701275, 14.70012287, 
         14.84741207, 15.61780065, 18.77192444, 22.9364153, 
         26.97581017, 30.26524343, 32.38373522, 33.37336815, 
         33.94616802, 33.91859929, 33.17270773, 31.43103978, 
         27.89686684, 23.94509292, 21.52756873, 20.77745354, 
         18.91836891, 17.03432652, 15.98249117, 15.62786054]]
edemandcurtdata =[
        [5.307235503, 4.895112175, 4.714461378, 4.469588753, 
         4.478136009, 4.587489228, 4.764915236, 4.690040646, 
         4.578926317, 4.607573714, 4.737316678, 5.096778890, 
         5.617895387, 6.325493913, 7.226313651, 8.184122924, 
         9.152295353, 9.794294472, 9.902982382, 10.14570254, 
         9.729884792, 8.348313208, 6.855580333, 5.887885360], 
        [5.229010890, 4.828565681, 4.609679220, 4.582119798, 
         4.885257784, 4.997820137, 4.903307209, 4.443508664, 
         4.106283407, 4.011238233, 4.240292172, 4.709193701, 
         5.345228526, 6.248732979, 7.463680032, 8.839115124, 
         10.28944046, 11.15921769, 11.12580356, 11.05012277, 
         10.27766059, 8.708652062, 7.216811482, 6.194514258],
        [5.572310021, 5.166315358, 4.897256816, 4.819008722, 
         4.952070401, 5.176349774, 5.122201498, 4.714750983, 
         4.310126072, 4.269346581, 4.631783113, 5.347975858, 
         6.207123809, 7.027879553, 7.95541338, 9.070470248, 
         10.38082252, 11.3022903, 11.35772223, 11.23394356, 
         10.45031203, 8.927820301, 7.373597276, 6.428734122]]
edemandcurtdata = np.array(edemandcurtdata)
edemandcurtdata += 8
def edemandcrit(t,n):
    return edemandcritdata[n-1][t-1]
def edemandcurt(t,n):
    return edemandcurtdata[n-1][t-1]
def edemand(t,n):
    return edemandcritdata[n-1][t-1] + edemandcurtdata[n-1][t-1]
# ranges of demand reduction (Table 2)
dmincoef = 0.1
dmaxcoef = 0.5
def dmin(t, n):
    return dmincoef * edemandcurt(t,n)
def dmax(t, n):
    return dmaxcoef * edemandcurt(t,n)
# price elasticities (Table 2)
elasticity_off_peak = -0.3
elasticity_mid_peak = -0.5
elasticity_on_peak = -0.7
def elasticity(t):
    assert t>=1 and t<=24
    if t <= 12:
        return elasticity_off_peak
    if t >= 17 and t <= 21:
        return elasticity_on_peak
    return elasticity_mid_peak
# energy consumption
def econscrit(t,n):
    return edemandcrit(t,n)
def econscurt(t,n,retprice):
    return edemandcurt(t,n) * (1 + elasticity(t) * ((retprice - wholeprice(t)) / wholeprice(t)))
def econs(t,n,retprice):
    return econscrit(t,n) + econscurt(t,n,retprice)
# dissatisfaction cost
def phi(t,n,retprice):
    return ((alphan[n-1] / 2) * (edemandcurt(t,n) - econscurt(t,n,retprice))**2) + (betan[n-1] * (edemandcurt(t,n) - econscurt(t,n,retprice)))


# OBJECTIVE FUNCTIONS
def cuobj(t,n,retprice):
    return retprice * econs(t,n,retprice) + phi(t,n,retprice)
#def cuobjfn(n):
    # minimize
#    return sum([cuobj(t+1,n) for t in range(ntimeslots)])
#def cuobjf():
    # minimize
#    return sum([cuobjfn(n+1) for n in range(ncustomers)])
def spobj(t,n,retprice):
    return (retprice - wholeprice(t)) * econs(t,n,retprice)
#def spobjf():
    # maximize
#    return sum([[spobj(t+1,n+1) for t in range(ntimeslots)] for n in range(ncustomers)])
def obj(t,n,retprice):
    # maximize
    return rho * spobj(t,n,retprice) - (1 - rho) * cuobj(t,n,retprice)
#def objf():
    # maximize
#    return rho * spobjf() - (1 - rho) * cuobjf()




# VISUALISE INPUT DATA

# Wholesale pricing
def showwholepricing():
    import matplotlib.pyplot as plt
    plt.plot(list(range(1,25)), wholepricedata, "o-")
    plt.title("Price data on June 22, 2017")
    plt.xlabel("Time slot")
    plt.ylabel("Wholesale price (ȼ/kWh)")
    plt.xlim(0.3,24.7)
    plt.xticks(list(range(1,25)))
    plt.yticks(list(range(0,7)))
    plt.show()
    return

# show customer energy demand
def showdemand(no=1):
    import matplotlib.pyplot as plt
    trange = list(range(1,25))
    p1 = plt.bar(trange, edemandcritdata[no-1], color="blue")
    p2 = plt.bar(trange, edemandcurtdata[no-1], color="red", bottom=edemandcritdata[no-1])
    plt.title("Customer " + str(no))
    plt.xlabel("Time slot")
    plt.ylabel("Energy demand (kWh)")
    plt.xlim(0.3,24.7)
    plt.xticks(trange)
    if no == 1:
        plt.yticks(list(range(0,35,5)))
    else:
        plt.yticks(list(range(0,50,5)))
    plt.legend((p1[0], p2[0]), ("Critical load", "Curtailable load"))
    plt.show()
    return





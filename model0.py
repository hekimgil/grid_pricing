#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hakan Hekimgil, Jafar Chaab
"""

# this is the action/output
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
        1.90, 1.80, 1.75, 1.70, 1.70, 2.00, 
        1.90, 1.80, 1.75, 1.70, 1.70, 2.00, 
        1.90, 1.80, 1.75, 1.70, 1.70, 2.00, 
        1.90, 1.80, 1.75, 1.70, 1.70, 2.00]
def wholeprice(t):
    return wholepricedata[t-1]

# CUSTOMER PARAMETERS
# customers' dissatisfaction related parameters (Table 2)
alpha = [0.8, 0.5, 0.3]
beta = [0.1, 0.1, 0.1]
# customer demand data (currentle from Fig. 5)
edemandcritdata =[
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50],
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50],
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50]]
edemandcurtdata =[
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50],
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50],
        [14.50, 14.25, 14.50, 14.25, 14.10, 13.70, 
         13.70, 14.10, 14.90, 16.00, 16.70, 17.20, 
         18.00, 18.10, 18.10, 18.20, 18.00, 17.50, 
         17.00, 17.10, 16.50, 16.00, 15.00, 14.50]]
def edemandcrit(t,n):
    return edemandcritdata[n-1,t-1]
def edemandcurt(t,n):
    return edemandcurtdata[n-1,t-1]
def edemand(t,n):
    return edemandcritdata[n-1,t-1] + edemandcurtdata[n-1,t-1]
# ranges of demand reduction
dmincoef = 0.1
dmaxcoef = 0.5
def dmin(t, n):
    return dmincoef * edemandcurt(t,n)
def dmax(t, n):
    return dmaxcoef * edemandcurt(t,n)
# price elasticities
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
def econscurt(t,n):
    return edemandcurt(t,n) * (1 + elasticity(t) * ((retailprice(t,n) - wholeprice(t)) / wholeprice(t)))
def econs(t,n):
    return econscrit(t,n) + econscurt(t,n)
# dissatisfaction cost
def phi(t,n):
    return (beta[n-1] + (alpha[n-1] / 2) * (edemand(t,n) - econs(t,n)))


# OBJECTIVE FUNCTIONS
def cuobj(t,n):
    return retailprice(t,n) * econs(t,n) + phi(t,n)
def cuobjfn(n):
    # minimize
    return sum([cuobj(t+1,n) for t in range(ntimeslots)])
def cuobjf():
    # minimize
    return sum([cuobjfn(n+1) for n in range(ncustomers)])
def spobj(t,n):
    return (retailprice(t,n) - wholeprice(t)) * econs(t,n)
def spobjf():
    # maximize
    return sum([[spobj(t+1,n+1) for t in range(ntimeslots)] for n in range(ncustomers)])
def objf():
    # maximize
    return rho * spobjf() - (1 - rho) * cuobjf()


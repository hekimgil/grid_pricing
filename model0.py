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
        1.90, 1.80, 1.70, 1.60, 1.60, 1.90,
        2.00, 2.21, 2.31, 2.60, 2.91, 3.30,
        3.40, 3.61, 3.50, 5.41, 5.51, 5.00,
        4.00, 3.21, 3.10, 3.60, 3.00, 2.60]
def wholeprice(t):
    return wholepricedata[t-1]

# CUSTOMER PARAMETERS
# customers' dissatisfaction related parameters (Table 2)
alpha = [0.8, 0.5, 0.3]
beta = [0.1, 0.1, 0.1]
# customer demand data (currentle from Fig. 5)
edemandcritdata =[
        [14.21, 13.94, 14.14, 13.80, 13.49, 12.32,
         12.25, 13.04, 14.66, 16.68, 17.75, 18.05,
         18.19, 18.57, 18.57, 18.57, 18.36, 17.95,
         17.51, 17.78, 17.06, 15.96, 15.14, 14.66],
        [14.15, 13.89, 13.74, 14.09, 14.71, 15.43,
         18.55, 22.91, 26.86, 29.57, 31.37, 31.78,
         32.39, 33.26, 33.16, 31.32, 27.63, 23.42,
         21.12, 20.65, 18.76, 16.86, 15.58, 15.12],
        [14.78, 14.52, 14.42, 14.67, 14.78, 15.60,
         18.79, 22.91, 26.93, 30.22, 32.33, 33.36,
         33.93, 33.88, 33.16, 31.41, 27.85, 23.89,
         21.47, 20.75, 18.84, 16.99, 15.91, 15.60]]
edemandcurtdata =[
        [19.53, 18.84, 18.81, 18.23, 17.99, 16.92,
         17.06, 17.71, 19.22, 21.28, 22.48, 23.17,
         23.82, 24.89, 25.85, 26.81, 27.56, 27.73,
         27.39, 27.94, 26.81, 24.34, 21.97, 20.56],
        [19.42, 18.71, 18.35, 18.71, 19.63, 20.45,
         23.53, 27.32, 30.96, 33.62, 35.57, 36.49,
         37.72, 39.57, 40.59, 40.18, 37.93, 34.60,
         32.24, 31.73, 29.06, 25.58, 22.86, 21.27],
        [20.39, 19.72, 19.31, 19.51, 19.77, 20.80,
         23.89, 27.65, 31.25, 34.55, 37.02, 38.72,
         40.16, 40.93, 41.09, 40.47, 38.26, 35.22,
         32.85, 31.97, 29.35, 25.95, 23.32, 22.04]]
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
    return ((alpha[n-1] / 2) * (edemand(t,n) - econs(t,n))**2) + (beta[n-1] * (edemand(t,n) - econs(t,n)))


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


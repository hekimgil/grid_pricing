#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses the model parameterized by t
author: Hakan Hekimgil, Jafar Chaab
"""

import numpy as np
import modelt

customer = 1

# model setup
ntimeslots = modelt.ntimeslots
actions = np.round(np.arange(2.4, 8.3, 0.1), 1)
nactions = len(actions)
#minprice = modelt.k1 * min(modelt.wholepricedata)
#possibleactions = {t:[a for a in range(nactions) if nactions[a] >= modelt.k1 * modelt.wholeprice(t) and nactions[a] <= modelt.k2 * modelt.wholeprice(t)] for t in range(1,ntimeslots+1)}
epsilon = 0.5
discount = 0.9
alpha = 0.1

def reward(t,n,price):
    return modelt.obj(t,n,price)

def futurerewards(t,n,greedy = True):
    if t >= ntimeslots:
        return 0.0
    else:
        if greedy and np.random.random() <= epsilon:
            randomaction = np.random.randint(nactions)
            randomprice = actions[randomaction]
            return reward(t,n,randomprice) + discount * futurerewards(t+1,n)
        else:
            bestaction = np.argmax(qmatrix[t-1,:])
            bestprice = actions[bestaction]
            return reward(t,n,bestprice) + discount * futurerewards(t+1,n)

# initialization
timeslot = 1
#i = 1
qmatrix = np.zeros([ntimeslots+1,nactions]) # one extra row
#qmatrix = np.full([ntimeslots+1,nactions], -np.inf) # one extra row


# Q-Learning loop
for i in range(100):
    for t in range(1,ntimeslots+1):
        qprev = qmatrix.copy()
        for action in range(nactions):
            # IMPORTANT REMINDER:
            # The functions use t as it is so reward(t,n,p) refers to time t
            # Arrays use t with 0-index addressing so qmatrix[t-1:] refers to time t
            qmatrix[t-1,action] = (
                    (1 - alpha ) * qprev[t-1,action] + 
                    alpha * (
                            reward(t,1,actions[action]) + 
                            discount * np.max(qprev[t,:])))
#    totalreward = 0
#    action = np.argmax(qmatrix[t-1,:])
#    aprice = actions[action]
#    reward = modelt.obj(timeslot,1,aprice)

bestpolicy = [actions[x] for x in np.argmax(qmatrix[:-1,:], axis=1)]
print(bestpolicy)


# VISUALISE INPUT AND OUTPUT DATA

# Combined plot
def plotresults():
    import matplotlib.pyplot as plt
    trange = list(range(1,25))
    barw = 0.25
    trange1 = [t-barw for t in trange]
    fig, ax1 = plt.subplots()
    p1 = plt.bar(trange1, [modelt.edemandcurt(t,customer) for t in trange], width=barw, color="blue")
    p2 = plt.bar(trange, [modelt.econscurt(t,customer,bestpolicy[t-1]) for t in trange], width=barw, color="red")
    plt.title("Customer {:}".format(customer))
    plt.xlabel("Time slot")
    ax1.set_ylabel("Electricity (kWh)")
    plt.yticks(list(range(0,14,2)))
    plt.xlim(0.3,24.7)
    plt.xticks(list(range(1,25)))
    ax2 = ax1.twinx()
    p3 = plt.plot(trange, modelt.wholepricedata, "o-g")
    p4 = plt.plot(trange, bestpolicy, "o-r")
    ax2.set_ylabel("Price (ȼ/kWh)")
    plt.yticks(list(range(0,9)))
    plt.legend((p1[0], p2[0], p3[0], p4[0]), 
               ("Energy demand", "Energy consumption", "Wholesale price", "Retail price"), 
               loc=2)
    fig.tight_layout()
    plt.show()
    return


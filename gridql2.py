#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses the t2 model (parameterized by t, and with modified demand)
author: Hakan Hekimgil, Jafar Chaab
"""

import numpy as np
import modelt2

customer = 1

# model setup
ntimeslots = modelt2.ntimeslots
actions = np.round(np.arange(2.4, 25.1, 0.1), 1)
nactions = len(actions)
#minprice = modelt2.k1 * min(modelt2.wholepricedata)
#possibleactions = {t:[a for a in range(nactions) if nactions[a] >= modelt2.k1 * modelt2.wholeprice(t) and nactions[a] <= modelt2.k2 * modelt2.wholeprice(t)] for t in range(1,ntimeslots+1)}
epsilon = 0.5   # rate for exploration
discount = 0.9  # discount rate for future rewards
alpha = 0.1     # learning rate

def reward(t,n,price):
    return modelt2.obj(t,n,price)

def takeaction(state,n,greedy = True):
    if greedy and np.random.random() <= epsilon:
        price = 0
        while price < modelt2.wholeprice(t):
            randomaction = np.random.randint(nactions)
            price = actions[randomaction]
        #randomprice = actions[randomaction]
        return randomaction
    else:
        bestaction = np.argmax(qmatrix[state,:])
        #bestprice = actions[bestaction]
        return bestaction

# initialization
timeslot = 1
iterations = 0
qmatrix = np.zeros([ntimeslots+1,nactions]) # one extra row
qprev = 1000*np.ones([ntimeslots+1,nactions]) # one extra row
delta = 0.01
convergence = []
#qmatrix = np.full([ntimeslots+1,nactions], -np.inf) # one extra row


# Q-Learning loop
while np.max(np.abs(qmatrix-qprev)) > delta:
    iterations += 1
    if iterations % 100 == 0:
        print("iteration {:,}; delta: {:}...".format(iterations, np.max(np.abs(qmatrix-qprev))))
    qprev = qmatrix.copy()
    for t in range(1,ntimeslots+1):
        qtemp = qmatrix.copy()
        for action in range(nactions):
            # IMPORTANT REMINDER:
            # The functions use t as it is so reward(t,n,p) refers to time t
            # Arrays use t with 0-index addressing so qmatrix[t-1:] refers to time t
            qmatrix[t-1,action] = (
                    (1 - alpha ) * qtemp[t-1,action] + 
                    alpha * (
                            reward(t,customer,actions[action]) + 
                            discount * np.max(qtemp[t,:])))
    convergence.append(np.max(np.abs(qmatrix-qprev)))
print("finished at iteration {:,}, with a delta of {:}...".format(iterations, np.max(np.abs(qmatrix-qprev))))

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
    p1 = plt.bar(trange1, [modelt2.edemandcurt(t,customer) for t in trange], width=barw, color="blue")
    p2 = plt.bar(trange, [modelt2.econscurt(t,customer,bestpolicy[t-1]) for t in trange], width=barw, color="red")
    plt.title("Customer {:}".format(customer))
    plt.xlabel("Time slot")
    ax1.set_ylabel("Electricity (kWh)")
    plt.yticks(list(range(0,22,2)))
    plt.xlim(0.3,24.7)
    plt.xticks(list(range(1,25)))
    ax2 = ax1.twinx()
    p3 = plt.plot(trange, modelt2.wholepricedata, "o-g")
    p4 = plt.plot(trange, bestpolicy, "o-r")
    ax2.set_ylabel("Price (ȼ/kWh)")
    plt.yticks(list(range(0,25)))
    plt.legend((p1[0], p2[0], p3[0], p4[0]), 
               ("Energy demand", "Energy consumption", "Wholesale price", "Retail price"), 
               loc=2)
    fig.tight_layout()
    plt.show()
    return

def plotconvergence():
    import matplotlib.pyplot as plt
    plt.plot(convergence)
    plt.show()
    return

plotresults()
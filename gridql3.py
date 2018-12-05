#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses the t3 model (parameterized by t, and with dissatisfaction multipliers)
author: Hakan Hekimgil, Jafar Chaab
"""

import numpy as np
import modelt3

customer = 1

# model setup
ntimeslots = modelt3.ntimeslots
actions = np.round(np.arange(2.4, 8.3, 0.1), 1)
nactions = len(actions)
#minprice = modelt2.k1 * min(modelt2.wholepricedata)
#possibleactions = {t:[a for a in range(nactions) if nactions[a] >= modelt2.k1 * modelt2.wholeprice(t) and nactions[a] <= modelt2.k2 * modelt2.wholeprice(t)] for t in range(1,ntimeslots+1)}
epsilon = 0.5   # rate for exploration
discount = 0.9  # discount rate for future rewards
alpha = 0.1     # learning rate

def reward(t,n,price):
    return modelt3.obj(t,n,price)

def takeaction(t, n, greedy = True):
    if greedy and np.random.random() <= epsilon:
        price = 0
        while price < modelt3.wholeprice(t):
            randomaction = np.random.randint(nactions)
            price = actions[randomaction]
        return randomaction
    else:
        bestaction = np.argmax(qmatrix[t-1,:])
        #bestprice = actions[bestaction]
        return bestaction

# initialization
timeslot = 1
iterations = 0
qmatrix = np.zeros([ntimeslots+1,nactions]) # one extra row
qprev = 1000*np.ones([ntimeslots+1,nactions]) # one extra row
delta = 0.01
convergence = []
qconvergence = []
#qmatrix = np.full([ntimeslots+1,nactions], -np.inf) # one extra row


# Q-Learning loop
while np.max(np.abs(qmatrix-qprev)) > delta:
    iterations += 1
    if iterations % 100 == 0:
        print("iteration {:,}; delta: {:}...".format(iterations, np.max(np.abs(qmatrix-qprev))))
    qprev = qmatrix.copy()
    for t in range(1,ntimeslots+1):
        for action in range(nactions):
            # IMPORTANT REMINDER:
            # The functions use t as it is so reward(t,n,p) refers to time t
            # Arrays use t with 0-index addressing so qmatrix[t-1:] refers to time t
            qmatrix[t-1,action] = (
                    (1 - alpha ) * qprev[t-1,action] + 
                    alpha * (
                            reward(t,customer,actions[action]) + 
                            discount * np.max(qprev[t,:])))
    convergence.append(np.max(np.abs(qmatrix-qprev)))
    qconvergence.append(np.mean(np.abs(qmatrix)))
print("finished at iteration {:,}, with a delta of {:}...".format(iterations, np.max(np.abs(qmatrix-qprev))))
#    totalreward = 0
#    action = np.argmax(qmatrix[t-1,:])
#    aprice = actions[action]
#    reward = modelt2.obj(timeslot,1,aprice)

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
    p1 = plt.bar(trange1, [modelt3.edemandcurt(t,customer) for t in trange], width=barw, color="blue")
    p2 = plt.bar(trange, [modelt3.econscurt(t,customer,bestpolicy[t-1]) for t in trange], width=barw, color="red")
    plt.title("Customer {:}".format(customer))
    plt.xlabel("Time slot")
    ax1.set_ylabel("Electricity (kWh)")
    plt.yticks(list(range(0,14,2)))
    plt.xlim(0.3,24.7)
    plt.xticks(list(range(1,25)))
    ax2 = ax1.twinx()
    p3 = plt.plot(trange, modelt3.wholepricedata, "o-g")
    p4 = plt.plot(trange, bestpolicy, "o-r")
    ax2.set_ylabel("Price (ȼ/kWh)")
    plt.yticks(list(range(0,10)))
    plt.legend((p1[0], p2[0], p3[0], p4[0]), 
               ("Energy demand", "Energy consumption", "Wholesale price", "Retail price"), 
               loc=2)
    fig.tight_layout()
    plt.savefig("CU"+str(customer)+"graph.png")
    plt.savefig("CU"+str(customer)+"graph.pdf")
    plt.show()
    return

def plotconvergence():
    import matplotlib.pyplot as plt
    plt.plot(convergence)
    plt.show()
    plt.plot(qconvergence)
    plt.show()
    return

print("Dissatisfaction multipliers (mul0, mul1, mul2): ", modelt3.dmul[customer-1])
plotresults()
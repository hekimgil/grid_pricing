#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses the model parameterized by t
author: Hakan Hekimgil, Jafar Chaab
"""

import numpy as np
import modelt

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

print([ actions[x] for x in np.argmax(qmatrix[:-1,:], axis=1)])
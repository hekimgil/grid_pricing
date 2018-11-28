#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses the t2 model (parameterized by t, and modified demand)
author: Hakan Hekimgil, Jafar Chaab
"""

import numpy as np
import modelt2

customer = 1

# model setup
ntimeslots = modelt2.ntimeslots
actions = np.round(np.arange(2.4, 22.1, 0.1), 1)
nactions = len(actions)
#minprice = modelt2.k1 * min(modelt2.wholepricedata)
#possibleactions = {t:[a for a in range(nactions) if nactions[a] >= modelt2.k1 * modelt2.wholeprice(t) and nactions[a] <= modelt2.k2 * modelt2.wholeprice(t)] for t in range(1,ntimeslots+1)}
epsilon = 0.5
discount = 0.9
alpha = 0.1

def reward(t,n,price):
    return modelt2.obj(t,n,price)

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
            return reward(t,n,bestprice) + discount* futurerewards(t+1,n)

# initialization
timeslot = 1
iterations = 0
qmatrix = np.zeros([ntimeslots+1,nactions]) # one extra row
qprev = 1000*np.ones([ntimeslots+1,nactions]) # one extra row
delta = 0.01
#qmatrix = np.full([ntimeslots+1,nactions], -np.inf) # one extra row


# Q-Learning loop
while np.max(np.abs(qmatrix-qprev)) > delta:
    iterations += 1
    if iterations % 100 == 0:
        print("iteration {:,}; delta: {:}...".format(iterations, np.max(np.abs(qmatrix-qprev))))
    qprev = qmatrix.copy()
    for t in range(1,ntimeslots+1):
        for action in range(nactions):
            qmatrix[t-1,action] = (
                    (1 - alpha ) * qmatrix[t-1,action] + 
                    alpha * (
                            reward(t,customer,actions[action]) + 
                            discount * np.max(qmatrix[t,:])))
print("finished at iteration {:,}, with a delta of {:}...".format(iterations, np.max(np.abs(qmatrix-qprev))))
#    totalreward = 0
#    action = np.argmax(qmatrix[t-1,:])
#    aprice = actions[action]
#    reward = modelt2.obj(timeslot,1,aprice)

print([ actions[x] for x in np.argmax(qmatrix[:-1,:], axis=1)])
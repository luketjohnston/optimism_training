from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def readdata(filename):
  f = open(filename, 'r')
  lines = f.readlines()
  x = []
  for l in lines:
    split = l.split()
    x.append(float(split[0]))
  return x



order = readdata('save/order.txt')
policy = readdata('save/policy.txt')
entropy = readdata('save/entropy.txt')
value = readdata('save/value.txt')
total = readdata('save/total.txt')
total = np.array(total) / 4


#plt.plot(order, label='Order loss') 
#plt.plot(policy, label='policy loss') 
plt.plot(entropy, label='entropy loss') 
#plt.plot(value, label='value loss') 
#plt.plot(total, label='total loss / 4') 

plt.legend()
plt.show()

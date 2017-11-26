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


plt.plot(order, label='Order loss') 

plt.legend()
plt.show()

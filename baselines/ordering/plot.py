import matplotlib.pyplot as plt


def readdata(filename):
  f = open(filename, 'r')
  lines = f.readlines()
  x = []
  for l in lines:
    split = l.split()
    x.append(float(split[0]))
  return x



progress = readdata('save/progress.txt')


plt.plot(progress, label='Progress loss') 

plt.legend()
plt.show()

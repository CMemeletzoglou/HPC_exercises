import pandas as pd
import numpy as np
import sys
import math
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


df = pd.read_csv(sys.argv[1], header=None, sep=' ')


print(df)

x = df.values[:, 0]
y = df.values[:, 1]
c = df.values[:, 2]

plt.scatter(x, y, c=c)
plt.colorbar()
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('f(x)=x[0]*x[0]+x[1]*x[1]')

plt.show()



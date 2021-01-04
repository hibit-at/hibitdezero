import numpy as np
from dezero.core import *
from dezero import functions as F
import matplotlib.pyplot as plt
import dezero.layers as L

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)

L1 = L.Linear(1,10)
L2 = L.Linear(10,1)

def predict(x):
    y = L1(x)
    y = F.sigmoid(y)
    y = L2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    L1.cleargrad()
    L2.cleargrad()
    loss.backward()

    for L in [L1,L2]:
        for p in L.params():
            p.data -= lr*p.grad.data
    
    if i%1000 == 0:
        print(loss)

plt.scatter(x,y)
plt.scatter(x,y_pred.data,color='r')
plt.show()
